# Compaction And Context Views

This page defines the proposed durability boundary for conversation compaction.
It is an architectural contract, not a released public API.

## Ownership

Compaction crosses three layers, but each layer has one responsibility:

- `Agent` decides **when** to check context pressure and coordinates the
  operation at a safe turn boundary.
- lifecycle hooks decide policy, such as declining a compaction or supplying a
  custom portable summary.
- `Model` decides **how** to count, prepare, and compact model-visible context.
  Provider-native endpoints and wire formats stay below this boundary.

The checkpoint store only persists the resulting operation. It does not call a
model, choose a strategy, or mutate old history.

## Context Capacity

Every chat model should expose an effective context capacity:

```text
context_capacity: int | None
```

Resolution should follow this order:

1. explicit model configuration
2. provider/model profile, including `models.dev`
3. provider metadata
4. unknown (`None`)

Unknown must remain distinct from unlimited. Automatic compaction is disabled
when capacity cannot be established, unless the caller supplies an override.

The trigger must reserve room for output and protocol overhead:

```text
estimated_input + reserved_output + safety_margin >= context_capacity
```

Token estimation belongs to `Model` because tokenizers and provider payloads
differ. The Agent asks for the estimate at a checkpoint; it does not count
`ChatMessages` itself.

Compaction must run before the model-visible window exceeds the limit. This is
also required by OpenAI's standalone endpoint: the input sent to
`/responses/compact` must still fit in the model context window.

## Append-Only Operation

Compaction appends one operation to the interaction timeline. It never changes
or disables earlier items:

```python
{
    "type": "compaction",
    "item_id": "cmp_01...",
    "reason": "threshold",  # manual | threshold | overflow
    "compacted_through_item_id": "itm_01...",
    "views": [
        {
            "format": "messages",
            "items": [
                {"role": "system", "content": "Portable summary ..."},
                {"role": "user", "content": "Retained recent input"},
            ],
        }
    ],
    "metadata": {
        "provider": "openai",
        "model_id": "gpt-5.6",
        "api_mode": "responses",
        "input_tokens_before": 118400,
    },
}
```

`views` is a small list of complete model-visible windows, not a list of
patches. A generic compactor normally writes one `messages` view. A native
provider compactor may write a provider-bound view instead:

```python
{
    "format": "provider",
    "provider": "openai",
    "api_mode": "responses",
    "items": [...],  # exact /responses/compact output
}
```

Provider items are opaque. The framework must preserve their order and content
without pruning or trying to extract a human summary. OpenAI documents the
returned output as the canonical next context window and notes that it may
contain retained prior items in addition to an encrypted compaction item.

A compaction operation may contain both a portable `messages` view and a native
view when a strategy can produce both without a second lossy conversion. It
must not invent a portable summary by decoding opaque provider state.

## More Than One Compaction

Only the newest compatible compaction on the selected branch defines the base
context. Context projection is:

```text
newest compatible compaction view
  + timeline items appended after compacted_through_item_id
```

Older compaction operations remain available for audit and forks, but they do
not stack into the active prompt. This keeps the prompt bounded even after many
compactions and removes the need for an `active` flag.

The operation stores a complete view so projection never needs to read through
an earlier compaction. Retained head or tail items are materialized inside that
view. This mirrors the useful property of Pi's `retainedTail`: every compaction
is a self-contained context checkpoint rather than a pointer into mutable
history.

Logical self-containment can duplicate retained payloads in serialized
snapshots. Physical payload deduplication by immutable item identity may be
added inside a store later; it must not change this logical contract.

## Portability

The model selects the newest view compatible with its provider and API mode. A
`messages` view is the portable fallback. A provider view is usable only by the
same provider protocol that produced it.

If no compatible view exists, the runtime must not silently drop the
compaction. It should either:

- build a portable view from the append-only source history before switching
  providers;
- fork before the provider-bound compaction; or
- reject the switch with an actionable error.

This makes the real portability limit explicit. In particular, OpenAI's
encrypted compaction item is continuation state, not provider-neutral summary
text.

## Operation Boundaries

The future operation should emit:

```text
compaction.start
compaction.end
```

and accept a typed `before_compaction` hook containing at least:

```text
reason
model identity
estimated input tokens
context capacity
source boundary
```

The hook may decline or provide a complete portable view. Provider-native
compaction still executes inside `Model`; the Agent only coordinates the
operation and persists its result.

For automatic compaction, the operation belongs to the current run and occurs
between turns. Manual compaction can later become its own resumable operation.
In both cases, the durable sequence is:

```text
compaction.start
prepare or request compacted view
append compaction item
checkpoint save
compaction.end
```

If generation fails before the compaction item is appended, the previous view
remains authoritative. If persistence succeeds but event delivery is
interrupted, replay discovers the appended item and does not compact the same
boundary again.

## OpenAI Endpoint Constraint

OpenAI supports automatic server-managed compaction for stored response chains
and a stateless, ZDR-friendly `/responses/compact` endpoint. msgFlux manages its
own provider-neutral timeline, so the standalone endpoint is the relevant
native primitive. Its complete returned output must be stored as one provider
view and sent back as-is on the next Responses request.

This endpoint does not remove the need for threshold estimation: calling it
after the current window already exceeds the context capacity is too late.

## Design References

- [OpenAI standalone compact endpoint](https://developers.openai.com/api/docs/guides/compaction#standalone-compact-endpoint)
- [Pi harness v2 design](https://github.com/earendil-works/pi/blob/harness-v2/j4/packages/agent/docs/harness-v2.md)
