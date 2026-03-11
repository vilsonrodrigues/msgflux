"""Demo: Durable Inline DSL v2 — delta pattern, checkpoint, resume.

Cenários:
1. Delta pattern — módulos retornam dicts em vez de mutar
2. Mixed — módulos legacy + delta no mesmo pipeline
3. Checkpoint per step — estado salvo após cada step
4. Resume from crash — simula crash e retoma do checkpoint
5. While loop resume — crash mid-loop, retoma da iteração correta
6. LLM real — pipeline com Groq + checkpoint
"""

import msgflux as mf
import msgflux.nn as nn
import msgflux.nn.functional as F
from msgflux.dotdict import DELETE

mf.load_dotenv()

groq = mf.Model.chat_completion("groq/llama-3.3-70b-versatile")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _header(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def _show_store(store, ns, sid, rid):
    state = store.load_state(ns, sid, rid)
    if state is None:
        print("  Store: (vazio)")
        return
    print(f"  Store: status={state['status']}")
    cursor = state.get("cursor", {})
    print(f"  Cursor: step_index={cursor.get('step_index')}, "
          f"frames={cursor.get('frames', [])}")
    snap = state.get("message_snapshot", {})
    keys = [k for k in snap.keys() if not k.startswith("_")]
    print(f"  Snapshot keys: {keys}")
    if state.get("error"):
        print(f"  Error: {state['error']}")


# ===================================================================
# CENÁRIO 1: Delta pattern — módulos retornam dicts
# ===================================================================
def scenario_1():
    _header("CENÁRIO 1: Delta pattern — módulos retornam dicts")

    def extract(msg):
        """Retorna delta em vez de mutar msg."""
        return {"topic": "fotossíntese", "source": "user"}

    def enrich(msg):
        topic = msg.get("topic", "unknown")
        return {"facts": f"3 fatos sobre {topic}", "enriched": True}

    def summarize(msg):
        return {"summary": f"Resumo de {msg['topic']}: {msg['facts']}"}

    modules = {"extract": extract, "enrich": enrich, "summarize": summarize}
    msg = mf.dotdict({"question": "Como funciona a fotossíntese?"})

    print("  Expression: extract -> enrich -> summarize")
    print(f"  Input: {msg.to_dict()}\n")

    result = F.inline("extract -> enrich -> summarize", modules, msg)

    print(f"  question: {result['question']}")
    print(f"  topic:    {result['topic']}")
    print(f"  facts:    {result['facts']}")
    print(f"  summary:  {result['summary']}")
    print(f"  enriched: {result['enriched']}")


# ===================================================================
# CENÁRIO 2: Mixed — legacy + delta
# ===================================================================
def scenario_2():
    _header("CENÁRIO 2: Mixed — legacy (in-place) + delta")

    def legacy_prep(msg):
        """Legacy: muta in-place, retorna msg."""
        msg["counter"] = 0
        msg["items"] = []
        return msg

    def delta_add_item(msg):
        """Delta: retorna dict com mudanças."""
        items = list(msg.get("items", []))
        items.append(f"item_{msg['counter']}")
        return {"items": items, "counter": msg["counter"] + 1}

    def legacy_finalize(msg):
        """Legacy: muta in-place."""
        msg["status"] = "done"
        msg["total"] = len(msg["items"])

    modules = {
        "prep": legacy_prep,
        "add": delta_add_item,
        "finalize": legacy_finalize,
    }
    msg = mf.dotdict()

    result = F.inline("prep -> add -> add -> add -> finalize", modules, msg)

    print(f"  counter: {result['counter']}")
    print(f"  items:   {result['items']}")
    print(f"  status:  {result['status']}")
    print(f"  total:   {result['total']}")


# ===================================================================
# CENÁRIO 3: Checkpoint per step
# ===================================================================
def scenario_3():
    _header("CENÁRIO 3: Checkpoint per step")

    store = mf.InMemoryCheckpointStore()

    def step_a(msg):
        return {"a": "done"}

    def step_b(msg):
        return {"b": "done"}

    def step_c(msg):
        return {"c": "done"}

    modules = {"a": step_a, "b": step_b, "c": step_c}
    msg = mf.dotdict()

    result = F.inline(
        "a -> b -> c",
        modules,
        msg,
        store=store,
        namespace="demo",
        session_id="s1",
        run_id="run1",
    )

    print(f"  Result: {result.to_dict()}")
    _show_store(store, "demo", "s1", "run1")


# ===================================================================
# CENÁRIO 4: Resume from crash
# ===================================================================
def scenario_4():
    _header("CENÁRIO 4: Resume from crash")

    store = mf.InMemoryCheckpointStore()
    call_log = []

    def step_a(msg):
        call_log.append("a")
        return {"a": "done"}

    def step_b_crash(msg):
        call_log.append("b_crash")
        raise ConnectionError("Simulando crash no step B!")

    def step_b_ok(msg):
        call_log.append("b_ok")
        return {"b": "recovered"}

    def step_c(msg):
        call_log.append("c")
        return {"c": "done"}

    # RUN 1: vai crashar no step B
    print("  RUN 1: a -> b(crash) -> c")
    modules_crash = {"a": step_a, "b": step_b_crash, "c": step_c}

    try:
        F.inline(
            "a -> b -> c",
            modules_crash,
            mf.dotdict(),
            store=store,
            namespace="demo",
            session_id="s1",
            run_id="resume1",
        )
    except ConnectionError as e:
        print(f"  Crash: {e}")

    print(f"  Modules executados: {call_log}")
    _show_store(store, "demo", "s1", "resume1")

    # RUN 2: resume — step A pula, step B roda (agora ok), step C roda
    print("\n  RUN 2: resume — a(skip) -> b(ok) -> c")
    call_log.clear()
    modules_ok = {"a": step_a, "b": step_b_ok, "c": step_c}

    result = F.inline(
        "a -> b -> c",
        modules_ok,
        mf.dotdict(),
        store=store,
        namespace="demo",
        session_id="s1",
        run_id="resume1",
    )

    print(f"  Modules executados: {call_log}")
    print(f"  Result: {result.to_dict()}")
    print(f"  step A pulou? {'a' not in call_log}")
    _show_store(store, "demo", "s1", "resume1")


# ===================================================================
# CENÁRIO 5: While loop resume
# ===================================================================
def scenario_5():
    _header("CENÁRIO 5: While loop — crash e resume mid-loop")

    store = mf.InMemoryCheckpointStore()
    call_count = {"n": 0}

    def init(msg):
        return {"counter": 0}

    def increment(msg):
        call_count["n"] += 1
        return {"counter": msg["counter"] + 1}

    def increment_crash_at_3(msg):
        call_count["n"] += 1
        new_val = msg["counter"] + 1
        if new_val == 3:
            raise RuntimeError("Crash na iteração 3!")
        return {"counter": new_val}

    def done(msg):
        return {"status": "completed"}

    expr = "init -> @{counter < 5}: increment; -> done"

    # RUN 1: crash na iteração 3
    print("  RUN 1: counter 0→1→2→crash!")
    modules_crash = {
        "init": init,
        "increment": increment_crash_at_3,
        "done": done,
    }

    try:
        F.inline(
            expr,
            modules_crash,
            mf.dotdict(),
            store=store,
            namespace="while_demo",
            session_id="s1",
            run_id="loop1",
        )
    except RuntimeError as e:
        print(f"  Crash: {e}")

    _show_store(store, "while_demo", "s1", "loop1")

    # RUN 2: resume — continua de counter=2
    print("\n  RUN 2: resume de counter=2, continua até 5")
    call_count["n"] = 0
    modules_ok = {"init": init, "increment": increment, "done": done}

    result = F.inline(
        expr,
        modules_ok,
        mf.dotdict(),
        store=store,
        namespace="while_demo",
        session_id="s1",
        run_id="loop1",
    )

    print(f"  counter final: {result['counter']}")
    print(f"  status: {result['status']}")
    print(f"  increment chamado {call_count['n']}x (deveria ser 3: 3,4,5)")
    _show_store(store, "while_demo", "s1", "loop1")


# ===================================================================
# CENÁRIO 6: LLM real — pipeline com Groq + checkpoint
# ===================================================================
def scenario_6():
    _header("CENÁRIO 6: LLM real — Groq pipeline + checkpoint")

    store = mf.InMemoryCheckpointStore()

    topic_extractor = nn.Agent(
        "topic_extractor",
        groq,
        system_message=(
            "Extraia o tema central da pergunta em 1-2 palavras. "
            "Responda APENAS com o tema."
        ),
    )

    enricher = nn.Agent(
        "enricher",
        groq,
        system_message=(
            "Gere 2 fatos curiosos sobre o tema. "
            "Responda em português, formato lista."
        ),
    )

    summarizer = nn.Agent(
        "summarizer",
        groq,
        system_message=(
            "Com base na pergunta e nos fatos, produza uma resposta "
            "em português em no máximo 2 frases."
        ),
    )

    def extract_topic(msg):
        topic = str(topic_extractor(msg["question"])).strip()
        return {"topic": topic}

    def enrich(msg):
        facts = str(enricher(f"Tema: {msg['topic']}")).strip()
        return {"facts": facts}

    def summarize(msg):
        prompt = (
            f"Pergunta: {msg['question']}\n"
            f"Tema: {msg['topic']}\n"
            f"Fatos: {msg['facts']}"
        )
        answer = str(summarizer(prompt)).strip()
        return {"answer": answer}

    modules = {
        "extract": extract_topic,
        "enrich": enrich,
        "summarize": summarize,
    }

    msg = mf.dotdict({"question": "Por que o céu é azul?"})

    print(f"  Pergunta: {msg['question']}")
    print("  Pipeline: extract -> enrich -> summarize\n")

    result = F.inline(
        "extract -> enrich -> summarize",
        modules,
        msg,
        store=store,
        namespace="llm_pipeline",
        session_id="demo",
        run_id="llm_run1",
    )

    print(f"  Tema:     {result['topic']}")
    print(f"  Fatos:    {result['facts'][:120]}...")
    print(f"  Resposta: {result['answer'][:200]}")
    print()
    _show_store(store, "llm_pipeline", "demo", "llm_run1")


# ===================================================================
# CENÁRIO 7: DELETE sentinel
# ===================================================================
def scenario_7():
    _header("CENÁRIO 7: DELETE sentinel — remover campos via delta")

    def setup(msg):
        return {"temp_data": "processando...", "result": 42, "debug": True}

    def cleanup(msg):
        """Remove campos temporários via DELETE."""
        return {"temp_data": DELETE, "debug": DELETE, "cleaned": True}

    modules = {"setup": setup, "cleanup": cleanup}
    msg = mf.dotdict()

    result = F.inline("setup -> cleanup", modules, msg)

    print(f"  result:    {result.get('result')}")
    print(f"  cleaned:   {result.get('cleaned')}")
    print(f"  temp_data: {result.get('temp_data', '(removido)')}")
    print(f"  debug:     {result.get('debug', '(removido)')}")
    print(f"  Keys:      {list(result.keys())}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  DEMO: Durable Inline DSL v2")
    print("=" * 60)

    scenario_1()  # Delta pattern
    scenario_2()  # Mixed legacy + delta
    scenario_3()  # Checkpoint per step
    scenario_4()  # Resume from crash
    scenario_5()  # While loop resume
    scenario_6()  # LLM real com Groq
    scenario_7()  # DELETE sentinel

    print(f"\n{'=' * 60}")
    print("  Demo concluída!")
    print(f"{'=' * 60}")
