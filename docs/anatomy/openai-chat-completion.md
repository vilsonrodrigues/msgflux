# OpenAI Chat Completion

`src/msgflux/models/providers/openai.py` contains several OpenAI-backed model
classes, but the chat architecture is split between
`OpenAICompatibleChatCompletion` and the concrete `OpenAIChatCompletion`
provider.

The compatible class owns the shared transport lifecycle. The concrete OpenAI
class declares OpenAI-only capabilities, supported API modes, and reasoning
codecs. This is where msgFlux translates the generic `Agent` contract into
either Chat Completions or Responses wire payloads.

That translation is not a thin pass-through. The provider is responsible for:

- building OpenAI request parameters
- adapting msgFlux output contracts to OpenAI constraints
- decoding provider responses into `ModelResponse`
- restoring provider transport payloads back to logical runtime shapes
- enforcing option combinations that should fail early

## Why This Provider Matters

This module is one of the main boundaries between msgFlux runtime semantics and
provider semantics.

Upstream code thinks in terms such as:

- `generation_schema`
- `typed_parser`
- `ToolFlowControl`
- `prefilling`
- `tool_definitions`

The wire API expects either:

- Chat Completions `messages` and `response_format`, or
- Responses `input` items and `text.format`
- `tools`
- `tool_choice`
- provider-specific payload shapes

`OpenAICompatibleChatCompletion` is the reusable adapter boundary.
`OpenAIChatCompletion` enables both `api_mode="chat_completions"` and
`api_mode="responses"` and defaults to Responses. Compatible third-party
providers inherit only the Chat Completions mode unless they explicitly
implement another protocol.

GPT-5.6 Responses can return multiple assistant message items distinguished by
`phase`. The parser selects `final_answer` for normal outputs and `commentary`
for `ToolFlowControl`, avoiding concatenation of independent structured JSON
documents. Each native message remains a distinct history item with its `phase`,
identity, status, and content, so manual replay does not replace it with a
synthetic assistant message. Streaming builds the same item incrementally.
Opaque reasoning items without summary text are reconstructed with an explicit
empty `summary` list before replay.

## Main Flow

The non-streaming path looks like this:

```text
Agent
  -> OpenAICompatibleChatCompletion.__call__(...)
  -> _validate_chat_completion_options(...)
  -> _build_generation_params(...)
  -> _generate(...)
     -> _prepare_generate_kwargs(...)
     -> _execute_model(...)
     -> _process_model_output(...)
  -> ModelResponse
```

The async path mirrors the same structure through `acall(...)` and
`_agenerate(...)`.

## Two Preparation Stages

There are two preparation steps, and they solve different problems.

### 1. `_build_generation_params(...)`

This method dispatches to the envelope builder selected by `api_mode`:

- converts canonical `ChatMessages` inside the Model
- injects `system_prompt`
- keeps `prefilling`
- expands `tool_definitions` into native `tools` and `tool_choice` when present

Chat Completions produces `messages`. Responses produces `input`, flattens
function definitions, and converts the remaining frontend parameters later in
`_adapt_responses_params(...)`.

### 2. `_prepare_generate_kwargs(...)`

This is where schema logic becomes provider-specific while the logical schema
remains API-neutral.

It decides how msgFlux output contracts should be exposed to OpenAI.

That includes:

- `typed_parser`
- canonical `generation_schema`
- flow-control metadata carried through `ToolDefinitions`
- the OpenAI `response_format`, later mapped to `text.format` for Responses
- transport normalization metadata

This is the method where logical schema and provider schema stop being the same
thing.

## Logical Schema vs Provider Schema

The provider follows the split documented in
[Logical vs Provider Schema](logical-vs-provider-schema.md).

At this layer:

- `generation_schema` is the canonical msgFlux runtime schema
- `transport_generation_schema` is the OpenAI-facing schema metadata

The transport metadata contains two pieces:

- `decoder_schema`
- `normalize`

Conceptually:

```text
generation_schema
  -> maybe lower for OpenAI
  -> response_format
  -> OpenAI returns payload
  -> decode with decoder_schema
  -> normalize transport payload
  -> validate against generation_schema
```

This keeps provider constraints from leaking into the runtime contract.

## Structured Output Branches

`_prepare_generate_kwargs(...)` has three main branches for non-streaming
structured output.

### 1. Typed Parser

If `typed_parser` is set, the provider does not build an OpenAI structured
output schema from `generation_schema`.

Instead:

- raw text is returned by the model
- the parser decodes that text
- optionally, msgFlux validates the parsed output against
  `generation_schema`

This branch is parser-oriented rather than provider-schema-oriented.

### 2. Plain `generation_schema`

If `generation_schema` is present and is not a `ToolFlowControl`, the provider
derives an OpenAI `response_format` from it.

If necessary, the schema is lowered first. This is where cases such as
`dict[K, V]` become provider-compatible transport shapes.

### 3. `ToolFlowControl`

If `generation_schema` is a `ToolFlowControl`, the provider asks the flow
control whether it wants to override the provider-facing schema.

That happens through:

- `build_provider_response_format(...)`
- `normalize_provider_response(...)`

This is how ReAct can send a provider-specific action schema while still
consuming a normalized runtime shape afterward.

## Response Decoding

For Chat Completions, `_process_completion_model_output(...)` converts the
choice into a `ModelResponse`. For Responses,
`_process_responses_model_output(...)` walks ordered output items and converts
messages, function calls, reasoning summaries, usage, and incomplete status.

The Responses path keeps summaries canonical and stores only provider-only
reasoning state under an identity containing `provider`, `api_mode`, and codec.
It reconstructs the complete native reasoning item only when a later Responses
request uses that same identity.

There are four major result shapes:

- native tool call payloads
- text completions
- structured outputs
- audio outputs

The structured path is the important one for the current architecture.

It does the following:

```text
OpenAI content string
  -> decode transport payload
  -> convert struct to dict when needed
  -> apply transport normalizer
  -> validate against canonical generation_schema
  -> return dotdict payload
```

That last validation step matters. It means the provider does not trust the
transport schema alone. The final runtime object is still checked against the
logical schema expected by msgFlux.

## Tool Calls And Structured Outputs

This provider handles two distinct tool-related modes:

### Native OpenAI Tool Calls

If OpenAI returns `tool_calls`, the provider builds a `ToolCallAggregator` and
returns a `ModelResponse` with `response_type="tool_call"`.

At that point the provider stops. The loop continues in `Agent`.

### Structured Tool Loops

If the agent is using a `ToolFlowControl` such as ReAct, the provider does not
rely on native `tool_calls`.

Instead it:

- receives a structured response payload
- decodes it
- normalizes it back to the logical flow shape
- returns that shape to `Agent`

So the provider supports both tool systems, but they are separate branches.

## Early Validation

`_validate_chat_completion_options(...)` exists to reject incompatible
combinations before an OpenAI request is made.

Today that includes:

- `prefilling` + `generation_schema`
- `stream=True` + `typed_parser`

The first rule is especially important because `prefilling` appends an
assistant message into the prompt, while `generation_schema` expects the model
to produce a structured payload from the start.

Conceptually:

```text
prefilling
  -> "continue from this assistant text"

generation_schema
  -> "start a strict structured output payload"
```

Those are conflicting instructions in this provider path, so the provider fails
fast instead of sending an ambiguous request.

## Streaming

The streaming path is intentionally simpler than the structured-output path.

In streaming mode, the selected API adapter:

- creates a `ModelStreamResponse`
- consumes Chat Completions chunks or typed Responses events
- aggregates text, reasoning, and native tool call deltas
- sets response metadata as the stream completes

It does not combine `stream=True` with typed-parser decoding, and structured
schema-heavy normalization is not the primary path there.

## ASCII Diagram

This is the provider's main decision tree:

```text
Agent params
  -> __call__ / acall
  -> validate options
  -> build generation params
  -> prepare generate kwargs
       |
       +--> typed_parser branch
       |
       +--> generation_schema branch
       |      |
       |      +--> lower schema for OpenAI if needed
       |      +--> build response_format
       |
       +--> ToolFlowControl branch
              |
              +--> build_provider_response_format(...)
              +--> keep normalize_provider_response(...)
  -> select api_mode
       |
       +--> chat_completions -> messages / response_format
       |
       +--> responses -> input / text.format
  -> execute OpenAI request
  -> process completion or Responses output
       |
       +--> tool_calls -> ToolCallAggregator
       |
       +--> text -> plain ModelResponse
       |
       +--> structured
              |
              +--> decode transport payload
              +--> normalize transport payload
              +--> validate against generation_schema
  -> ModelResponse
```

## Relationship To The Rest Of The System

This provider should be read together with:

- [Agent](agent.md)
- [ToolFlowControl](tool-flow-control.md)
- [Logical vs Provider Schema](logical-vs-provider-schema.md)
- [ReAct Provider Schemas](react-provider-schemas.md)

The design line is:

- `Agent` assembles the runtime contract
- `OpenAICompatibleChatCompletion` owns the common frontend and lifecycle
- `OpenAIChatCompletion` declares OpenAI modes and codecs
- the provider returns normalized output back to the runtime contract

## Why This Shape Matters

Without this adapter layer, provider restrictions would leak into signatures,
generation schemas, flow controls, and tool execution.

This module keeps those concerns localized:

- OpenAI-specific request formatting stays here
- OpenAI-specific transport schemas stay here
- final runtime validation still points back to msgFlux contracts

That balance is the main reason the newer transport-schema work is sustainable.
