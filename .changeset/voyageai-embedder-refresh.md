---
'@mastra/voyageai': minor
---

Refreshed the VoyageAI embedder with new models and reliable contextualized embeddings.

**New models**

- Added `voyage-4-nano`, the smallest open-weight model.
- Added `voyage-code-4`, the latest model for code retrieval.

**Contextualized embeddings are now a drop-in embedder**

Contextualized models (`voyage-context-3`, `voyage-context-4`) now return exactly one embedding per input, so they work anywhere a plain text embedder is expected, including Mastra Memory and `embedMany`. Each input string is embedded independently as its own document, with server-side auto-chunking so each input resolves to a single chunk. Queries skip auto-chunking, which the API does not allow for `inputType: 'query'`.

```typescript
import { voyageContextualizedEmbedding } from '@mastra/voyageai';

const contextual = voyageContextualizedEmbedding('voyage-context-4');

// One embedding per input
const { embeddings } = await contextual.doEmbed({
  values: ['First document...', 'Second document...'],
});
console.log(embeddings.length); // 2
```

**Token-aware batching**

Text and contextualized embedding requests are now split into batches by estimated token count against each model's per-request limit, not only by item count, which avoids oversized requests on long inputs.
