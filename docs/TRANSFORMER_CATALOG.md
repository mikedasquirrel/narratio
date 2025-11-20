# Transformer Catalog

> _Updated automatically via `narrative_optimization.src.transformers.registry`._

## Why this exists

We repeatedly hit `class not found` errors because nobody had a canonical index
of the 100+ transformers that live under `narrative_optimization/src/transformers`.
The new registry walks the entire package with the AST, records metadata for every
`*Transformer` subclass, and exposes it via a small CLI so new bots (and humans)
can answer, in seconds:

- Which transformers currently ship with the system?
- Where do they live?
- Are we referencing a transformer that has not been implemented yet?
- How are transformers distributed across semantic/temporal/meta categories?

## Quick usage

Run everything from the repo root.

```bash
python -m narrative_optimization.tools.list_transformers
```

Sample (truncated) output:

```
Class                         | Slug                       | Category | Module
------------------------------+----------------------------+----------+---------------------------------------------------------
ActantialStructureTransformer | actantial_structure        | core     | narrative_optimization.src.transformers.actantial_structure
AuthenticityTransformer       | authenticity               | core     | narrative_optimization.src.transformers.authenticity
...
```

### Filter / search

```bash
python -m narrative_optimization.tools.list_transformers --filter nominative
python -m narrative_optimization.tools.list_transformers --category semantic
```

### Validate references

```bash
python -m narrative_optimization.tools.list_transformers --check NarrativeMass foo_bar
```

Returns non-zero if any names are missing and prints fuzzy suggestions, making it
ideal for CI or quick smoke tests when editing configs.

### JSON / Markdown exports

```bash
python -m narrative_optimization.tools.list_transformers --format json > transformers.json
python -m narrative_optimization.tools.list_transformers --format markdown --summary
```

## Programmatic access

Import the registry directly when you need metadata:

```python
from narrative_optimization.src.transformers.registry import get_transformer_registry

registry = get_transformer_registry()
metadata = registry.resolve("narrative_potential")
print(metadata.class_name)  # NarrativePotentialTransformer
print(metadata.module_path) # narrative_optimization.src.transformers.narrative_potential
print(registry.summary_by_category())  # {'core': 23, 'semantic': 9, ...}
```

## Error handling improvements

- `from narrative_optimization.src import transformers` now exposes the constant
  `AVAILABLE_TRANSFORMERS` so you can introspect the full catalog without
  loading every module.
- When `TransformerFactory` cannot resolve a name it now surfaces fuzzy matches
  and reminds you to run the CLI above.

This eliminates silent failure modes and makes the transformer surface obvious
to every bot on day one.


