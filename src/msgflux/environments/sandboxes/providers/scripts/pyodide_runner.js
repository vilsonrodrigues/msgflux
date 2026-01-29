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

// JSON-RPC 2.0 error codes
const JSONRPC_ERRORS = {
  PARSE_ERROR: -32700,
  INVALID_REQUEST: -32600,
  METHOD_NOT_FOUND: -32601,
  INVALID_PARAMS: -32602,
  INTERNAL_ERROR: -32603,
  // Application-specific errors
  SYNTAX_ERROR: -32000,
  NAME_ERROR: -32001,
  TYPE_ERROR: -32002,
  VALUE_ERROR: -32003,
  RUNTIME_ERROR: -32004,
  EXECUTION_ERROR: -32005,
  TOOL_ERROR: -32006,
};

/**
 * Create a JSON-RPC 2.0 response
 */
function jsonrpcResult(result, id) {
  return JSON.stringify({ jsonrpc: "2.0", result, id });
}

/**
 * Create a JSON-RPC 2.0 error response
 */
function jsonrpcError(code, message, id, data = null) {
  const error = { code, message };
  if (data !== null) {
    error.data = data;
  }
  return JSON.stringify({ jsonrpc: "2.0", error, id });
}

/**
 * Create a JSON-RPC 2.0 request (for tool calls to host)
 */
function jsonrpcRequest(method, params, id) {
  return JSON.stringify({ jsonrpc: "2.0", method, params, id });
}

/**
 * Map Python exception types to JSON-RPC error codes
 */
function getErrorCode(errorType) {
  const mapping = {
    SyntaxError: JSONRPC_ERRORS.SYNTAX_ERROR,
    NameError: JSONRPC_ERRORS.NAME_ERROR,
    TypeError: JSONRPC_ERRORS.TYPE_ERROR,
    ValueError: JSONRPC_ERRORS.VALUE_ERROR,
    RuntimeError: JSONRPC_ERRORS.RUNTIME_ERROR,
  };
  return mapping[errorType] || JSONRPC_ERRORS.EXECUTION_ERROR;
}

/**
 * Main PyodideRunner class
 */
class PyodideRunner {
  constructor() {
    this.pyodide = null;
    this.initialized = false;
    this.mountedFiles = new Map();
    this.registeredTools = new Map();
    this.toolCallCounter = 0;
    this.pendingToolCalls = new Map();
    this.stdinReader = null;
  }

  /**
   * Initialize Pyodide runtime
   */
  async initialize() {
    const { loadPyodide } = await import("npm:pyodide@0.26.4");

    this.pyodide = await loadPyodide({
      stdout: (text) => { this.currentStdout += text + "\n"; },
      stderr: (text) => { this.currentStderr += text + "\n"; },
    });

    // Load micropip for package installation
    await this.pyodide.loadPackage("micropip");

    // Setup Python environment with tool support
    this.pyodide.runPython(`
import sys
import io
import json

# Store for capturing output
class OutputCapture:
    def __init__(self):
        self.stdout = ""
        self.stderr = ""
        self.old_stdout = None
        self.old_stderr = None

    def start(self):
        self.stdout = ""
        self.stderr = ""
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

    def stop(self):
        if self.old_stdout:
            self.stdout = sys.stdout.getvalue()
            self.stderr = sys.stderr.getvalue()
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr
            self.old_stdout = None
            self.old_stderr = None
        return self.stdout, self.stderr

_output_capture = OutputCapture()

# Variable storage accessible from JS
_sandbox_vars = {}

def _get_var(name):
    return json.dumps(_sandbox_vars.get(name))

def _set_var(name, value_json):
    _sandbox_vars[name] = json.loads(value_json)

def _get_all_vars():
    # Return only JSON-serializable variables
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

# Tool bridge - will be populated by register_tool
_tool_bridge = None

def _set_tool_bridge(bridge):
    global _tool_bridge
    _tool_bridge = bridge

def _call_tool(name, args, kwargs):
    """Call a tool registered on the host."""
    if _tool_bridge is None:
        raise RuntimeError("Tool bridge not initialized")
    return _tool_bridge(name, args, kwargs)
`);

    // Create the tool bridge function
    const toolBridge = this.pyodide.globals.get("_set_tool_bridge");
    const self = this;

    // Create a JavaScript function that will be called from Python
    const bridgeFunction = (name, args, kwargs) => {
      // This will be called synchronously from Python
      // We need to return a promise that resolves when the tool call completes
      return self._callHostTool(name, args.toJs(), kwargs.toJs());
    };

    toolBridge(bridgeFunction);

    this.initialized = true;
    return { status: "ready", version: this.pyodide.version };
  }

  /**
   * Call a tool on the host via JSON-RPC
   */
  async _callHostTool(name, args, kwargs) {
    this.toolCallCounter++;
    const callId = `tool_${this.toolCallCounter}`;

    // Send tool call request to host
    const request = jsonrpcRequest("tool_call", {
      tool_name: name,
      args: args,
      kwargs: kwargs,
    }, callId);

    console.log(request);

    // Wait for response from host
    // The host will send a response with the same ID
    return new Promise((resolve, reject) => {
      this.pendingToolCalls.set(callId, { resolve, reject });
    });
  }

  /**
   * Handle a tool call response from the host
   */
  handleToolResponse(response) {
    const callId = response.id;
    const pending = this.pendingToolCalls.get(callId);

    if (!pending) {
      console.error(`No pending tool call for ID: ${callId}`);
      return;
    }

    this.pendingToolCalls.delete(callId);

    if (response.error) {
      pending.reject(new Error(response.error.message));
    } else {
      pending.resolve(response.result);
    }
  }

  /**
   * Register a tool that can be called from Python
   */
  registerTool(name, parameters = []) {
    if (!this.initialized) {
      throw new Error("Pyodide not initialized");
    }

    this.registeredTools.set(name, { name, parameters });

    // Build Python function signature
    const sigParts = parameters.map(p => {
      let part = p.name;
      if (p.type) part += `: ${p.type}`;
      if (p.default !== undefined) {
        if (typeof p.default === 'string') {
          part += ` = "${p.default}"`;
        } else if (p.default === null) {
          part += ` = None`;
        } else {
          part += ` = ${JSON.stringify(p.default)}`;
        }
      }
      return part;
    });
    const signature = sigParts.join(', ');

    // Build args list for the call
    const argNames = parameters.map(p => p.name).join(', ');

    // Create Python wrapper function
    const wrapperCode = `
def ${name}(${signature}):
    """Tool function: ${name}"""
    import json
    from pyodide.ffi import run_sync
    args = [${argNames}]
    kwargs = {}
    result = run_sync(_call_tool("${name}", args, kwargs))
    # Convert JsProxy to Python if needed
    if hasattr(result, 'to_py'):
        return result.to_py()
    return result

# Also set in __main__
import __main__
setattr(__main__, "${name}", ${name})
`;

    this.pyodide.runPython(wrapperCode);
    return { success: true, tool: name };
  }

  /**
   * Execute Python code
   */
  async execute(code) {
    if (!this.initialized) {
      throw new Error("Pyodide not initialized");
    }

    const startTime = performance.now();
    this.currentStdout = "";
    this.currentStderr = "";

    try {
      // Start output capture
      this.pyodide.runPython("_output_capture.start()");

      // Load any packages from imports
      await this.pyodide.loadPackagesFromImports(code);

      // Execute the code
      const result = await this.pyodide.runPythonAsync(code);

      // Stop capture and get output
      const [stdout, stderr] = this.pyodide.runPython("_output_capture.stop()").toJs();

      // Update _sandbox_vars with new local variables
      this.pyodide.runPython(`
# Update sandbox vars with new variables from execution
import __main__
for name in dir(__main__):
    if not name.startswith('_'):
        try:
            val = getattr(__main__, name)
            if not callable(val) and not isinstance(val, type):
                json.dumps(val)  # Check if serializable
                _sandbox_vars[name] = val
        except:
            pass
`);

      // Get variables
      const variables = JSON.parse(this.pyodide.runPython("_get_all_vars()"));

      // Convert result to JS
      let returnValue = null;
      if (result !== undefined && result !== null) {
        try {
          returnValue = result.toJs ? result.toJs() : result;
          // Try to make it JSON serializable
          JSON.stringify(returnValue);
        } catch {
          returnValue = String(result);
        }
      }

      return {
        success: true,
        output: stdout + this.currentStdout,
        stderr: stderr + this.currentStderr,
        return_value: returnValue,
        variables: variables,
        execution_time_ms: performance.now() - startTime,
      };

    } catch (error) {
      // Stop capture on error
      try {
        this.pyodide.runPython("_output_capture.stop()");
      } catch {}

      const errorType = error.type || error.name || "Error";
      const errorMessage = error.message || String(error);

      return {
        success: false,
        error: {
          type: errorType,
          message: errorMessage,
          code: getErrorCode(errorType),
        },
        output: this.currentStdout,
        stderr: this.currentStderr,
        execution_time_ms: performance.now() - startTime,
      };
    }
  }

  /**
   * Get a variable from the sandbox
   */
  getVariable(name) {
    if (!this.initialized) {
      throw new Error("Pyodide not initialized");
    }

    const result = this.pyodide.runPython(`_get_var(${JSON.stringify(name)})`);
    return JSON.parse(result);
  }

  /**
   * Set a variable in the sandbox
   */
  setVariable(name, value) {
    if (!this.initialized) {
      throw new Error("Pyodide not initialized");
    }

    const valueJson = JSON.stringify(value);
    this.pyodide.runPython(`_set_var(${JSON.stringify(name)}, ${JSON.stringify(valueJson)})`);

    // Also set in __main__ namespace
    this.pyodide.runPython(`
import __main__
setattr(__main__, ${JSON.stringify(name)}, json.loads(${JSON.stringify(valueJson)}))
`);

    return { success: true };
  }

  /**
   * Install a Python package via micropip
   */
  async installPackage(packageName) {
    if (!this.initialized) {
      throw new Error("Pyodide not initialized");
    }

    try {
      await this.pyodide.runPythonAsync(`
import micropip
await micropip.install(${JSON.stringify(packageName)})
`);
      return { success: true, package: packageName };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  /**
   * List installed packages
   */
  listPackages() {
    if (!this.initialized) {
      throw new Error("Pyodide not initialized");
    }

    const packages = this.pyodide.runPython(`
import micropip
list(micropip.list().keys())
`).toJs();

    return { packages: Array.from(packages) };
  }

  /**
   * Mount a file in the virtual filesystem
   */
  mountFile(path, content) {
    if (!this.initialized) {
      throw new Error("Pyodide not initialized");
    }

    // Decode base64 content
    const bytes = Uint8Array.from(atob(content), c => c.charCodeAt(0));

    // Ensure directory exists
    const dir = path.substring(0, path.lastIndexOf('/'));
    if (dir) {
      try {
        this.pyodide.FS.mkdirTree(dir);
      } catch {}
    }

    // Write file
    this.pyodide.FS.writeFile(path, bytes);
    this.mountedFiles.set(path, true);

    return { success: true, path: path };
  }

  /**
   * Reset the sandbox state
   */
  reset() {
    if (!this.initialized) {
      throw new Error("Pyodide not initialized");
    }

    // Clear variables
    this.pyodide.runPython("_clear_vars()");

    // Clear __main__ namespace (but keep registered tools)
    const toolNames = Array.from(this.registeredTools.keys());
    const keepNames = new Set(toolNames);

    this.pyodide.runPython(`
import __main__
_keep_names = ${JSON.stringify(toolNames)}
for name in list(dir(__main__)):
    if not name.startswith('_') and name not in _keep_names:
        try:
            delattr(__main__, name)
        except:
            pass
`);

    // Remove mounted files
    for (const path of this.mountedFiles.keys()) {
      try {
        this.pyodide.FS.unlink(path);
      } catch {}
    }
    this.mountedFiles.clear();

    return { success: true };
  }

  /**
   * List registered tools
   */
  listTools() {
    return {
      tools: Array.from(this.registeredTools.keys()),
    };
  }
}

/**
 * Handle a JSON-RPC request
 */
async function handleRequest(runner, request) {
  const { method, params = {}, id } = request;

  try {
    switch (method) {
      case "ping":
        return { status: "ready" };

      case "execute":
        return await runner.execute(params.code || "");

      case "get_variable":
        return { value: runner.getVariable(params.name) };

      case "set_variable":
        return runner.setVariable(params.name, params.value);

      case "register_tool":
        return runner.registerTool(params.name, params.parameters || []);

      case "list_tools":
        return runner.listTools();

      case "install_package":
        return await runner.installPackage(params.package);

      case "list_packages":
        return runner.listPackages();

      case "mount_file":
        return runner.mountFile(params.path, params.content);

      case "reset":
        return runner.reset();

      case "shutdown":
        return { status: "shutdown" };

      default:
        throw { code: JSONRPC_ERRORS.METHOD_NOT_FOUND, message: `Unknown method: ${method}` };
    }
  } catch (error) {
    if (error.code) {
      throw error;
    }
    throw { code: JSONRPC_ERRORS.INTERNAL_ERROR, message: error.message || String(error) };
  }
}

/**
 * Main entry point
 */
async function main() {
  const runner = new PyodideRunner();

  // Initialize Pyodide
  try {
    const initResult = await runner.initialize();
    console.log(jsonrpcResult(initResult, "init"));
  } catch (error) {
    console.log(jsonrpcError(JSONRPC_ERRORS.INTERNAL_ERROR, `Init failed: ${error.message}`, "init"));
    Deno.exit(1);
  }

  // Read from stdin line by line
  const decoder = new TextDecoder();
  const reader = Deno.stdin.readable.getReader();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Process complete lines
    let newlineIndex;
    while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
      const line = buffer.substring(0, newlineIndex).trim();
      buffer = buffer.substring(newlineIndex + 1);

      if (!line) continue;

      let request;
      let requestId = null;

      try {
        request = JSON.parse(line);
        requestId = request.id;
      } catch (error) {
        console.log(jsonrpcError(JSONRPC_ERRORS.PARSE_ERROR, "Invalid JSON", null));
        continue;
      }

      // Check if this is a tool call response from host
      if (request.result !== undefined || request.error !== undefined) {
        runner.handleToolResponse(request);
        continue;
      }

      // Validate JSON-RPC request
      if (!request.method) {
        console.log(jsonrpcError(JSONRPC_ERRORS.INVALID_REQUEST, "Missing method", requestId));
        continue;
      }

      // Handle shutdown
      if (request.method === "shutdown") {
        console.log(jsonrpcResult({ status: "shutdown" }, requestId));
        Deno.exit(0);
      }

      // Process request
      try {
        const result = await handleRequest(runner, request);
        console.log(jsonrpcResult(result, requestId));
      } catch (error) {
        const code = error.code || JSONRPC_ERRORS.INTERNAL_ERROR;
        const message = error.message || String(error);
        console.log(jsonrpcError(code, message, requestId));
      }
    }
  }
}

// Run
main().catch((error) => {
  console.error("Fatal error:", error);
  Deno.exit(1);
});
