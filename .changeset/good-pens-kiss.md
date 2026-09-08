---
'@mastra/voyageai': minor
---

Refreshed the VoyageAI by MongoDB embedder against the current model catalog.

**Added text embedding models**

- `voyage-4-nano`: the smallest open-weight model, available via `voyage.v4nano` (and `voyage.v4nanoV2`).
- `voyage-code-4`: the latest model for code retrieval, available via `voyage.code4` (and `voyage.code4V2`).

**Added reranker models**

- `rerank-3` and `rerank-3-lite` (in preview), available via `voyage.reranker3` and `voyage.reranker3lite`.

```typescript
import { voyage } from '@mastra/voyageai';

const { embeddings } = await voyage.code4.doEmbed({ values: ['function add(a, b) { return a + b }'] });
```

Also updated the package name and description to reflect that Voyage AI is now part of MongoDB.
