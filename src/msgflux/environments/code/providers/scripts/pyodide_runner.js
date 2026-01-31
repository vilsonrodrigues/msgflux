/**
 * Pyodide Runner for msgFlux Sandbox
 *
 * This Deno script provides a secure Python execution environment using Pyodide
 * (Python compiled to WebAssembly). Communication happens via JSON-RPC 2.0 over
 * stdin/stdout.
 *
 * Features:
 * - Secure code execution in WebAssembly sandbox
 * - Tool registration and execution via JSON-RPC bridge
 * - Variable injection and extraction
 * - Package installation via micropip
 *
 * Usage: deno run [permissions] pyodide_runner.js
 */

import pyodideModule from "npm:pyodide@0.26.4/pyodide.js";
import { readLines } from "https://deno.land/std@0.186.0/io/mod.ts";

// =============================================================================
// JSON-RPC 2.0 Constants and Helpers
// =============================================================================

const JSONRPC_PROTOCOL_ERRORS = {
  ParseError: -32700,
  InvalidRequest: -32600,
  MethodNotFound: -32601,
};

const JSONRPC_APP_ERRORS = {
  SyntaxError: -32000,
  NameError: -32001,
  TypeError: -32002,
  ValueError: -32003,
  AttributeError: -32004,
  IndexError: -32005,
  KeyError: -32006,
  RuntimeError: -32007,
  Unknown: -32099,
};

const jsonrpcRequest = (method, params, id) =>
  JSON.stringify({ jsonrpc: "2.0", method, params, id });

const jsonrpcResult = (result, id) =>
  JSON.stringify({ jsonrpc: "2.0", result, id });

const jsonrpcError = (code, message, id, data = null) => {
  const err = { code, message };
  if (data) err.data = data;
  return JSON.stringify({ jsonrpc: "2.0", error: err, id });
};

// =============================================================================
// Python Setup Code
// =============================================================================

const PYTHON_SETUP_CODE = `
import sys, io, json

# Output capture
class OutputCapture:
    def __init__(self):
        self.stdout = ""
        self.stderr = ""
        self._old_stdout = None
        self._old_stderr = None
        self._buf_stdout = None
        self._buf_stderr = None

    def start(self):
        self.stdout = ""
        self.stderr = ""
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        self._buf_stdout = io.StringIO()
        self._buf_stderr = io.StringIO()
        sys.stdout = self._buf_stdout
        sys.stderr = self._buf_stderr

    def stop(self):
        if self._old_stdout:
            self.stdout = self._buf_stdout.getvalue()
            self.stderr = self._buf_stderr.getvalue()
            sys.stdout = self._old_stdout
            sys.stderr = self._old_stderr
            self._old_stdout = None
            self._old_stderr = None
        return self.stdout, self.stderr

_output_capture = OutputCapture()

# Variable storage
_sandbox_vars = {}

def _get_var(name):
    return json.dumps(_sandbox_vars.get(name))

def _set_var(name, value_json):
    _sandbox_vars[name] = json.loads(value_json)

def _get_all_vars():
    result = {}
    for k, v in _sandbox_vars.items():
        try:
            json.dumps(v)
            result[k] = v
        except (TypeError, ValueError):
            result[k] = str(v)
    return json.dumps(result)

def _clear_vars():
    _sandbox_vars.clear()

def _last_exception_args():
    return json.dumps(sys.last_exc.args) if hasattr(sys, 'last_exc') and sys.last_exc else None
`;

// =============================================================================
// Tool Wrapper Generation
// =============================================================================

const toPythonLiteral = (value) => {
  if (value === null) return 'None';
  if (value === true) return 'True';
  if (value === false) return 'False';
  return JSON.stringify(value);
};

const makeToolWrapper = (toolName, parameters = []) => {
  const sigParts = parameters.map(p => {
    let part = p.name;
    if (p.type) part += `: ${p.type}`;
    if (p.default !== undefined) part += ` = ${toPythonLiteral(p.default)}`;
    return part;
  });
  const signature = sigParts.join(', ');
  const argNames = parameters.map(p => p.name);

  if (parameters.length === 0) {
    return `
import json
from pyodide.ffi import run_sync, JsProxy
def ${toolName}(*args, **kwargs):
    """Tool function: ${toolName}"""
    result = run_sync(_js_tool_call("${toolName}", json.dumps({"args": args, "kwargs": kwargs})))
    return result.to_py() if isinstance(result, JsProxy) else result
`;
  }

  return `
import json
from pyodide.ffi import run_sync, JsProxy
def ${toolName}(${signature}):
    """Tool function: ${toolName}"""
    _args = [${argNames.join(', ')}]
    result = run_sync(_js_tool_call("${toolName}", json.dumps({"args": _args, "kwargs": {}})))
    return result.to_py() if isinstance(result, JsProxy) else result
`;
};

// =============================================================================
// Global Error Handler
// =============================================================================

globalThis.addEventListener("unhandledrejection", (event) => {
  event.preventDefault();
  console.log(jsonrpcError(
    JSONRPC_APP_ERRORS.RuntimeError,
    `Unhandled async error: ${event.reason?.message || event.reason}`,
    null
  ));
});

// =============================================================================
// Initialize Pyodide
// =============================================================================

const pyodide = await pyodideModule.loadPyodide();
await pyodide.loadPackage("micropip");

// Run initial setup
pyodide.runPython(PYTHON_SETUP_CODE);

// =============================================================================
// Tool Call Bridge (shared stdin reader)
// =============================================================================

// The stdin reader is shared so tool calls can read responses during execution
const stdinReader = readLines(Deno.stdin);
let requestIdCounter = 0;

// This function is called from Python to invoke a host-side tool
async function toolCallBridge(name, argsJson) {
  const requestId = `tc_${Date.now()}_${++requestIdCounter}`;

  try {
    const parsedArgs = JSON.parse(argsJson);

    // Send tool call request to host
    console.log(jsonrpcRequest("tool_call", {
      tool_name: name,
      args: parsedArgs.args || [],
      kwargs: parsedArgs.kwargs || {}
    }, requestId));

    // Wait for response from host (using shared stdin reader)
    const { value: responseLine, done } = await stdinReader.next();
    if (done) {
      throw new Error("stdin closed while waiting for tool response");
    }

    const response = JSON.parse(responseLine);

    if (response.id !== requestId) {
      throw new Error(`Response ID mismatch: expected ${requestId}, got ${response.id}`);
    }

    if (response.error) {
      throw new Error(response.error.message || "Tool call failed");
    }

    // Return result
    const result = response.result;
    if (result && result.type === "json") {
      return JSON.parse(result.value);
    }
    return result?.value ?? result;

  } catch (error) {
    throw new Error(`Tool '${name}' error: ${error.message}`);
  }
}

// Expose bridge to Python
pyodide.globals.set("_js_tool_call", toolCallBridge);

// Track registered tools
const registeredTools = new Map();

// =============================================================================
// Request Handlers
// =============================================================================

async function handleExecute(params, requestId) {
  const code = params.code || "";
  const startTime = performance.now();
  let setupCompleted = false;

  try {
    // Load any required packages
    await pyodide.loadPackagesFromImports(code);

    // Start output capture
    pyodide.runPython("_output_capture.start()");
    setupCompleted = true;

    // Execute the code
    const result = await pyodide.runPythonAsync(code);

    // Stop capture and get output
    const [stdout, stderr] = pyodide.runPython("_output_capture.stop()").toJs();

    // Update sandbox vars with new variables
    pyodide.runPython(`
import __main__
for name in dir(__main__):
    if not name.startswith('_'):
        try:
            val = getattr(__main__, name)
            if not callable(val) and not isinstance(val, type):
                json.dumps(val)
                _sandbox_vars[name] = val
        except:
            pass
`);

    // Get variables
    const variables = JSON.parse(pyodide.runPython("_get_all_vars()"));

    // Convert result
    let returnValue = null;
    if (result !== null && result !== undefined) {
      try {
        returnValue = result.toJs ? result.toJs() : result;
        JSON.stringify(returnValue);
      } catch {
        returnValue = String(result);
      }
    }

    return {
      success: true,
      output: stdout,
      stderr: stderr,
      return_value: returnValue,
      variables: variables,
      execution_time_ms: performance.now() - startTime,
    };

  } catch (error) {
    // Stop capture on error
    if (setupCompleted) {
      try {
        pyodide.runPython("_output_capture.stop()");
      } catch {}
    }

    const errorType = error.type || error.name || "Error";
    const errorMessage = (error.message || "").trim();

    // Get error args for Python exceptions
    let errorArgs = [];
    if (errorType !== "SyntaxError") {
      try {
        const argsJson = pyodide.runPython("_last_exception_args()");
        if (argsJson) {
          errorArgs = JSON.parse(argsJson) || [];
        }
      } catch {}
    }

    const errorCode = JSONRPC_APP_ERRORS[errorType] || JSONRPC_APP_ERRORS.Unknown;
    throw {
      code: errorCode,
      message: errorMessage,
      data: { type: errorType, args: errorArgs }
    };
  }
}

function handleRegisterTool(params) {
  const { name, parameters = [] } = params;

  if (!name) {
    throw { code: JSONRPC_APP_ERRORS.ValueError, message: "Tool name required" };
  }

  // Generate and run the wrapper
  const wrapper = makeToolWrapper(name, parameters);
  pyodide.runPython(wrapper);

  // Store tool info
  registeredTools.set(name, { name, parameters });

  return { success: true, tool: name };
}

function handleGetVariable(params) {
  const result = pyodide.runPython(`_get_var(${JSON.stringify(params.name)})`);
  return { value: JSON.parse(result) };
}

function handleSetVariable(params) {
  const { name, value } = params;
  const valueJson = JSON.stringify(value);

  pyodide.runPython(`_set_var(${JSON.stringify(name)}, ${JSON.stringify(valueJson)})`);

  // Also set in __main__
  pyodide.runPython(`
import __main__
setattr(__main__, ${JSON.stringify(name)}, json.loads(${JSON.stringify(valueJson)}))
`);

  return { success: true };
}

async function handleInstallPackage(params) {
  const { package: packageName } = params;

  try {
    await pyodide.runPythonAsync(`
import micropip
await micropip.install(${JSON.stringify(packageName)})
`);
    return { success: true, package: packageName };
  } catch (error) {
    return { success: false, error: error.message };
  }
}

function handleListPackages() {
  const packages = pyodide.runPython(`
import micropip
list(micropip.list().keys())
`).toJs();

  return { packages: Array.from(packages) };
}

function handleMountFile(params) {
  const { path, content } = params;

  // Decode base64 content
  const bytes = Uint8Array.from(atob(content), c => c.charCodeAt(0));

  // Ensure directory exists
  const dir = path.substring(0, path.lastIndexOf('/'));
  if (dir) {
    const parts = dir.split('/').filter(Boolean);
    let cur = '';
    for (const part of parts) {
      cur += '/' + part;
      try {
        pyodide.FS.mkdir(cur);
      } catch (e) {
        if (!e.message?.includes('File exists')) {
          // Directory already exists, continue
        }
      }
    }
  }

  // Write file
  pyodide.FS.writeFile(path, bytes);

  return { success: true, path: path };
}

function handleReset() {
  // Clear variables
  pyodide.runPython("_clear_vars()");

  // Get registered tool names to preserve
  const toolNames = Array.from(registeredTools.keys());

  // Clear __main__ namespace except tools
  pyodide.runPython(`
import __main__
_keep_names = ${JSON.stringify(toolNames)}
for name in list(dir(__main__)):
    if not name.startswith('_') and name not in _keep_names:
        try:
            delattr(__main__, name)
        except:
            pass
`);

  return { success: true };
}

function handleListTools() {
  return { tools: Array.from(registeredTools.keys()) };
}

// =============================================================================
// Main Loop
// =============================================================================

// Send ready signal
console.log(jsonrpcResult({ status: "ready", version: pyodide.version }, "init"));

// Process requests
while (true) {
  const { value: line, done } = await stdinReader.next();
  if (done) break;

  let input;
  let requestId = null;

  try {
    input = JSON.parse(line);
    requestId = input.id;
  } catch (error) {
    console.log(jsonrpcError(JSONRPC_PROTOCOL_ERRORS.ParseError, "Invalid JSON: " + error.message, null));
    continue;
  }

  // Validate JSON-RPC format
  if (typeof input !== 'object' || input === null) {
    console.log(jsonrpcError(JSONRPC_PROTOCOL_ERRORS.InvalidRequest, "Invalid request format", requestId));
    continue;
  }

  const method = input.method;
  const params = input.params || {};

  // Handle shutdown
  if (method === "shutdown") {
    console.log(jsonrpcResult({ status: "shutdown" }, requestId));
    break;
  }

  try {
    let result;

    switch (method) {
      case "ping":
        result = { status: "ready" };
        break;

      case "execute":
        result = await handleExecute(params, requestId);
        break;

      case "register_tool":
        result = handleRegisterTool(params);
        break;

      case "get_variable":
        result = handleGetVariable(params);
        break;

      case "set_variable":
        result = handleSetVariable(params);
        break;

      case "install_package":
        result = await handleInstallPackage(params);
        break;

      case "list_packages":
        result = handleListPackages();
        break;

      case "mount_file":
        result = handleMountFile(params);
        break;

      case "reset":
        result = handleReset();
        break;

      case "list_tools":
        result = handleListTools();
        break;

      default:
        throw { code: JSONRPC_PROTOCOL_ERRORS.MethodNotFound, message: `Unknown method: ${method}` };
    }

    console.log(jsonrpcResult(result, requestId));

  } catch (error) {
    const code = error.code || JSONRPC_APP_ERRORS.Unknown;
    const message = error.message || String(error);
    const data = error.data || null;
    console.log(jsonrpcError(code, message, requestId, data));
  }
}
