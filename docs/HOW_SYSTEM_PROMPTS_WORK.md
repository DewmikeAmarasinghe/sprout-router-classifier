# HOW_SYSTEM_PROMPTS_WORK.md

## What is a "system prompt per API call"?

Every time we call the OpenAI API to generate training data, we send two messages:
1. A **system message** — context and instructions for the LLM
2. A **user message** — the specific request for this batch

The system message is what we call the "system prompt per API call".
It is rebuilt fresh for each combination of `(language, industry, scenario)`.

---

## What goes into each system prompt?

Every system prompt has two parts:

### Part 1 — Shared context (same across all calls)
This is injected by `PromptFactory` using the shared template:

```
You are generating synthetic training data for Sprout — an AI-powered customer
service chatbot platform built by hSenid Mobile (Sri Lanka).

Sprout is deployed across multiple channels: WhatsApp Business, Instagram DMs,
Facebook Messenger, Viber, website chat widgets, and mobile app embedded chats.
It serves Sri Lankan businesses in these sectors: ecommerce, healthcare, banking,
insurance, telecom, logistics, hospitality, and education.

The data you generate will train a binary router classifier:
  label=0 → route to gpt-4o-mini (fast, cheap — pure English, simple queries)
  label=1 → route to gpt-4o (complex, code-mixed, or sensitive queries)

CURRENT CONTEXT:
  Industry:  {industry.description}
  Language:  {language.description}
  Platform:  {platform_style}
```

### Part 2 — Scenario-specific instructions (built by the SectionBuilder)
Each scenario has its own `SectionBuilder` class that adds:

```
TASK: {scenario.description}

Why this scenario routes to {label}:
  {scenario.routing_reason}

LENGTH: Generate messages in these proportions:
  {length_distribution.to_prompt_str()}

GENERIC EXAMPLES for {scenario.key} in {industry.key} ({language.key}):
  {examples_from_store}

DO NOT generate (these belong to {anti_scenario.key}, not {scenario.key}):
  {anti_examples_from_store}

Previously generated in this session (DO NOT repeat or paraphrase):
  {rolling_context_window}

Return JSON: {{"prompts": ["message1", "message2", ...]}}
Generate exactly {n} messages.
```

---

## How is it generated in code?

```
PromptFactory
│
├── build_system_prompt(language, industry, scenario, n, already_generated)
│   │
│   ├── shared_context = SharedContextBuilder.build(language, industry, platform_style)
│   │
│   ├── section_builder = SECTION_BUILDERS[scenario]          ← factory lookup
│   │   (e.g. LocationProximityBuilder, ContinuationBuilder, ...)
│   │
│   ├── scenario_section = section_builder.build(
│   │       language, industry, examples, anti_examples, length_dist
│   │   )
│   │
│   └── return shared_context + "\n\n" + scenario_section
│
└── SECTION_BUILDERS: dict[ScenarioKey, SectionBuilder]
    ├── ScenarioKey.SIMPLE_TRANSACTIONAL  → SimpleTransactionalBuilder()
    ├── ScenarioKey.NAMED_LOCATION        → NamedLocationBuilder()
    ├── ScenarioKey.LOCATION_PROXIMITY    → LocationProximityBuilder()
    ├── ScenarioKey.LOCATION_RELATIVE     → LocationRelativeBuilder()
    ├── ScenarioKey.COMPLEX_TASK          → ComplexTaskBuilder()
    ├── ScenarioKey.SENSITIVE_CONTEXT     → SensitiveContextBuilder()
    ├── ScenarioKey.ESCALATION            → EscalationBuilder()
    ├── ScenarioKey.RESPONSE_LANGUAGE     → ResponseLanguageBuilder()
    └── ScenarioKey.CONTINUATION          → ContinuationBuilder()
```

---

## How many unique system prompts are there?

```
languages × industries × scenarios
    5      ×     8     ×     9     = 360 unique system prompts
```

Each of the 360 cells also has rotating:
- Platform style (8 options, rotated per call)
- Examples (rotated from ExampleStore)
- Rolling context (previous 30 outputs, changes every call)

So in practice every API call has a slightly different prompt, but the structure is one of 360 base templates.

---

## Where can you see the prompts in the Gradio UI?

In the **Data** tab → **System Prompt Preview** section:

```
┌─────────────────────────────────────────┐
│ Language:  [singlish_light ▼]           │
│ Industry:  [banking ▼]                  │
│ Scenario:  [location_proximity ▼]       │
│                                         │
│ [Preview System Prompt]                 │
│                                         │
│ ┌─────────────────────────────────────┐ │
│ │ You are generating synthetic        │ │
│ │ training data for Sprout...         │ │
│ │                                     │ │
│ │ TASK: User asks for nearest         │ │
│ │ location without naming one...      │ │
│ │ ...                                 │ │
│ └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

The panel calls `prompt_factory.build_preview(lang, ind, sc)` — no logic in the panel.

---

## How do examples get into the prompt?

`ExampleStore.get(language, industry, scenario)` is called during prompt building:

```
First call (cold):
  ExampleStore.get(singlish_light, banking, location_proximity)
  → generates 8 examples via API
  → saves to examples.json
  → returns examples

Subsequent calls:
  → loads from examples.json
  → returns cached examples

After "Regenerate" button in UI:
  ExampleStore.regenerate(singlish_light, banking, location_proximity)
  → generates fresh examples
  → overwrites cache
  → returns new examples
```

The examples.json structure:
```json
{
  "singlish_light": {
    "banking": {
      "location_proximity": [
        "mata lagin thiyen branch eka koheda?",
        "nearest ATM near my area please",
        "which branch should I go to from here?"
      ]
    }
  }
}
```

---

## How does rolling context prevent repetition?

```python
# In GeneratorService.generate_cell()
already_generated: deque[str] = deque(maxlen=30)

for call_index in range(n_calls):
    prompt = prompt_factory.build_system_prompt(
        language=cell.language,
        industry=cell.industry,
        scenario=cell.scenario,
        n=batch_size,
        already_generated=list(already_generated),   # ← passed to prompt
    )
    batch = api_client.generate(prompt)

    for message in batch.prompts:
        already_generated.append(message)   # ← running window grows
```

The prompt includes:
```
Previously generated in this session (DO NOT repeat or paraphrase these):
  - "mata lagin thiyen branch eka koheda?"
  - "nearest ATM near my area please"
  - "which branch should I go to from here?"
  ... (up to 30 most recent)
```

This costs ~500 tokens per call in context but saves all the embedding + dedup compute.
For 60k rows / 50 per call = 1,200 calls × 30 context items = very manageable.

---

## Why NOT a 4th hierarchy level for platform?

Considered: `language → industry → scenario → platform`

Decision: NOT a separate level.

Reason: The routing signal we are training is independent of platform.
"mata lagin thiyen branch eka koheda?" routes to gpt-4o whether it arrives via WhatsApp
or a website widget. The response strategy changes by platform — not the routing decision.

Platform style IS varied — but as a rotation variable per call (one of 8 styles),
not a full hierarchy level. Adding it as a level would:
- Multiply cell count by 8× (360 → 2,880)
- Require 8× more API calls for the same number of rows per cell
- Add no meaningful signal diversity for the routing classifier

Platform style affects: punctuation, emoji use, formality, sentence length.
These are already covered by the length distribution and language instruction.
