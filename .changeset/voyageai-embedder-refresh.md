---
'@mastra/voyageai': minor
---

Refreshed the VoyageAI embedder with new models and reliable contextualized embeddings.

**New models**

- Added `voyage-4-nano`, the smallest open-weight model.
- Added `voyage-code-4`, the latest model for code retrieval.

**Contextualized embeddings are now a drop-in embedder**

Contextualized models (`voyage-context-3`, `voyage-context-4`) now return exactly one embedding per input, so they work anywhere a plain text embedder is expected, including Mastra Memory and `embedMany`. Each input string is embedded independently as its own document, with server-side auto-chunking so each input resolves to a single chunk. Queries skip auto-chunking, which the API does not allow for `inputType: 'query'`. An input longer than the per-chunk window (32,000 tokens) is averaged across its auto-chunks into one vector, with a warning, rather than silently dropping the extra chunks.

```typescript
import { voyageContextualizedEmbedding } from '@mastra/voyageai';

const contextual = voyageContextualizedEmbedding('voyage-context-4');

// One embedding per input
const { embeddings } = await contextual.doEmbed({
  values: ['First document...', 'Second document...'],
});
console.log(embeddings.length); // 2
```

**Breaking:** the contextualized embedding API changed to make one-embedding-per-input the default. This package is pre-1.0, so it ships as a minor, but existing contextualized callers must migrate:

- `values` is now a flat `string[]` (each string embedded independently) instead of `string[][]` (chunks of one document embedded together).
- Per-call `inputType` and `outputDimension` arguments on `doEmbed` are replaced by `providerOptions.voyage`.
- The exported class `VoyageContextualizedEmbeddingModel` is split into `VoyageContextualizedEmbeddingModelV2` and `VoyageContextualizedEmbeddingModelV3`. `voyageContextualizedEmbedding()` / `createVoyageContextualizedEmbedding()` now return the V3 model (use `...V2` for the AI SDK v5 model).
- The `embedQuery`, `embedDocumentChunks`, and `doEmbedGrouped` helpers, the `chunkCounts` result field, and the `maxTotalChunks` property were removed.

```typescript
// Before
const model = voyageContextualizedEmbedding('voyage-context-3');
const q = await model.embedQuery('What was the revenue?');
const docs = await model.embedDocumentChunks(['chunk 1', 'chunk 2']);
const { embeddings } = await model.doEmbed({
  values: [['chunk 1', 'chunk 2']],
  inputType: 'document',
  outputDimension: 512,
});

// After
const model = voyageContextualizedEmbedding('voyage-context-4');
const { embeddings: q } = await model.doEmbed({
  values: ['What was the revenue?'],
  providerOptions: { voyage: { inputType: 'query' } },
});
const { embeddings } = await model.doEmbed({
  values: ['chunk 1', 'chunk 2'], // each string embedded independently
  providerOptions: { voyage: { outputDimension: 512 } },
});
```

Cross-input contextualization (embedding several chunks of the *same* document together) is intentionally no longer exposed, since generic `embedMany` callers pass unrelated texts. Chunk your document and embed those chunks in your own request if you need it.

**Token-aware batching**

Text and contextualized embedding requests are now split into batches by estimated token count against each model's per-request limit, not only by item count, which avoids oversized requests on long inputs.
