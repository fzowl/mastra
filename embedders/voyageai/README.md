# @mastra/voyageai

VoyageAI embeddings integration for Mastra. Provides text, multimodal, and contextualized chunk embeddings using the official VoyageAI TypeScript SDK.

## Installation

```bash
npm install @mastra/voyageai
# or
pnpm add @mastra/voyageai
```

## Configuration

Set your VoyageAI API key:

```bash
export VOYAGE_API_KEY=your-api-key
```

Or pass it directly in the configuration.

## Usage

### Text Embeddings

```typescript
import { voyage, voyageEmbedding } from '@mastra/voyageai';

// Use default model (voyage-3.5)
const result = await voyage.doEmbed({ values: ['Hello world'] });
console.log(result.embeddings); // [[0.1, 0.2, ...]]

// Use specific model with options
const model = voyageEmbedding({
  model: 'voyage-3-large',
  inputType: 'query',
  outputDimension: 512,
});
const queryResult = await model.doEmbed({ values: ['search query'] });
```

### Pre-configured Models

```typescript
import { voyage } from '@mastra/voyageai';

// Voyage-4 series (highest throughput)
await voyage.v4large.doEmbed({ values: ['...'] }); // voyage-4-large (120k batch tokens)
await voyage.v4.doEmbed({ values: ['...'] }); // voyage-4 (320k batch tokens)
await voyage.v4lite.doEmbed({ values: ['...'] }); // voyage-4-lite (1M batch tokens)
await voyage.v4nano.doEmbed({ values: ['...'] }); // voyage-4-nano (open-weight)
await voyage.code4.doEmbed({ values: ['...'] }); // voyage-code-4 (code retrieval)

// Voyage-3 series
await voyage.large.doEmbed({ values: ['...'] }); // voyage-3-large
await voyage.v35.doEmbed({ values: ['...'] }); // voyage-3.5
await voyage.v35lite.doEmbed({ values: ['...'] }); // voyage-3.5-lite
await voyage.code.doEmbed({ values: ['...'] }); // voyage-code-3
await voyage.finance.doEmbed({ values: ['...'] }); // voyage-finance-2
await voyage.law.doEmbed({ values: ['...'] }); // voyage-law-2
```

### With Mastra Memory

```typescript
import { Memory } from '@mastra/memory';
import { PgVector } from '@mastra/pg';
import { voyage } from '@mastra/voyageai';

const memory = new Memory({
  vector: new PgVector(connectionString),
  embedder: voyage,
  options: {
    semanticRecall: { topK: 5 },
  },
});
```

### VoyageAI-Specific Options

```typescript
import { voyageEmbedding } from '@mastra/voyageai';

const model = voyageEmbedding({
  model: 'voyage-3.5',
  inputType: 'query', // 'query' | 'document' for retrieval optimization
  outputDimension: 512, // 256 | 512 | 1024 | 2048
  outputDtype: 'float', // 'float' | 'int8' | 'uint8' | 'binary' | 'ubinary'
  truncation: true, // Handle long inputs
  baseUrl: 'https://ai.mongodb.com/v1', // Optional: custom endpoint (e.g. MongoDB-hosted Voyage)
});
```

### Runtime Options Override

```typescript
const result = await model.doEmbed({
  values: ['query text'],
  providerOptions: {
    voyage: {
      inputType: 'query',
      outputDimension: 256,
    },
  },
});
```

## Multimodal Embeddings

Embed interleaved text + images + video (3.5 only):

```typescript
import { voyageMultimodalEmbedding } from '@mastra/voyageai';

const multimodal = voyageMultimodalEmbedding('voyage-multimodal-3.5');

const result = await multimodal.doEmbed({
  values: [
    {
      content: [
        { type: 'text', text: 'A photo of a cat' },
        { type: 'image_url', image_url: 'https://example.com/cat.jpg' },
        // video_url supported on voyage-multimodal-3.5
      ],
    },
  ],
});

// Use with vector store directly
await vectorStore.upsert({
  vectors: result.embeddings,
  metadata: [{ description: 'cat photo' }],
});
```

### Content Types

- `{ type: 'text', text: string }` - Text content
- `{ type: 'image_url', image_url: string }` - Image from URL
- `{ type: 'image_base64', image_base64: string }` - Base64-encoded image
- `{ type: 'video_url', video_url: string }` - Video from URL (3.5 only)

## Contextualized Chunk Embeddings

Contextualized models produce embeddings that account for the surrounding document, which improves retrieval over plain chunk embeddings. This integration exposes them as a standard text embedder: **each input string is embedded independently as its own document**, so you get exactly one embedding per input.

```typescript
import { voyageContextualizedEmbedding } from '@mastra/voyageai';

const contextual = voyageContextualizedEmbedding('voyage-context-4');

// Each string is embedded independently (one embedding per input)
const result = await contextual.doEmbed({
  values: ['First document...', 'Second document...'],
});
console.log(result.embeddings.length); // 2

// Embed a query for retrieval
const query = await contextual.doEmbed({
  values: ['What was the revenue?'],
  providerOptions: { voyage: { inputType: 'query' } },
});
```

Because inputs are embedded independently, this model is a drop-in embedder for Mastra Memory and any `embedMany` caller. Documents are sent as a flat list with server-side auto-chunking (chunk size 32,000 tokens) so each input resolves to a single chunk. Queries skip auto-chunking, which the API does not allow for `inputType: 'query'`.

An input longer than the 32,000-token per-chunk window is split into multiple chunks server-side. To keep the one-embedding-per-input contract, those chunk embeddings are averaged into a single vector and a warning is logged; pre-chunk or shorten such inputs if you need each chunk embedded separately.

Cross-input contextualization (embedding several chunks of the *same* document together so each vector reflects the others) is intentionally not used here, since generic callers pass unrelated texts. To use it, send one document's chunks together in your own request.

## Available Models

### Text Embedding Models

| Model              | Use Case                                | Dimensions        | Batch Tokens |
| ------------------ | --------------------------------------- | ----------------- | ------------ |
| `voyage-4-large`   | Best quality, highest batch capacity    | 256/512/1024/2048 | 120k         |
| `voyage-4`         | Balanced quality/speed, high throughput | 256/512/1024/2048 | 320k         |
| `voyage-4-lite`    | Maximum throughput                      | 256/512/1024/2048 | 1M           |
| `voyage-4-nano`    | Smallest, open-weight                   | 256/512/1024/2048 | 1M           |
| `voyage-code-4`    | Latest code retrieval                   | 256/512/1024/2048 | 120k         |
| `voyage-3-large`   | Best quality, multilingual              | 256/512/1024/2048 | 120k         |
| `voyage-3.5`       | Balanced quality/speed                  | 256/512/1024/2048 | 320k         |
| `voyage-3.5-lite`  | Lowest latency/cost                     | 256/512/1024/2048 | 1M           |
| `voyage-code-3`    | Code retrieval                          | 256/512/1024/2048 | 32k          |
| `voyage-finance-2` | Finance domain                          | 1024              | 32k          |
| `voyage-law-2`     | Legal domain                            | 1024              | 32k          |

### Multimodal Models

| Model                   | Capabilities          |
| ----------------------- | --------------------- |
| `voyage-multimodal-3`   | Text + images         |
| `voyage-multimodal-3.5` | Text + images + video |

### Contextualized Models

| Model              | Use Case                                              |
| ------------------ | ----------------------------------------------------- |
| `voyage-context-4` | Document-aware embeddings, best quality (recommended) |
| `voyage-context-3` | Document-aware embeddings (previous generation)       |

### Reranker Models

| Model             | Context Length | Description                             |
| ----------------- | -------------- | --------------------------------------- |
| `rerank-2.5`      | 32000          | Best quality with instruction-following |
| `rerank-2.5-lite` | 32000          | Optimized for latency and quality       |
| `rerank-2`        | 16000          | Second-gen with multilingual support    |
| `rerank-2-lite`   | 8000           | Second-gen, latency-optimized           |
| `rerank-1`        | 8000           | First-gen, quality-focused              |
| `rerank-lite-1`   | 4000           | First-gen, latency-optimized            |

## Reranking

VoyageAI rerankers implement the `RelevanceScoreProvider` interface for use with Mastra's reranking system.

### Basic Usage

```typescript
import { voyage, voyageReranker, createVoyageReranker } from '@mastra/voyageai';

// Use pre-configured reranker (rerank-2.5)
const defaultReranker = voyage.reranker;

// Or create with specific model
const liteReranker = createVoyageReranker('rerank-2.5-lite');

// Or with full config
const customReranker = createVoyageReranker({
  model: 'rerank-2.5',
  truncation: true,
});
```

### Get Relevance Score

```typescript
// Score a single document against a query
const score = await reranker.getRelevanceScore(
  'What is machine learning?',
  'Machine learning is a subset of artificial intelligence...',
);
console.log(score); // 0.85
```

### Rerank Multiple Documents

```typescript
// Rerank multiple documents efficiently in one API call
const results = await reranker.rerankDocuments(
  'What is the capital of France?',
  ['Paris is the capital of France.', 'London is the capital of England.', 'Berlin is the capital of Germany.'],
  2, // topK - optional
);

// Results sorted by relevance
console.log(results);
// [
//   { document: 'Paris is the capital of France.', index: 0, score: 0.95 },
//   { document: 'Berlin is the capital of Germany.', index: 2, score: 0.32 },
// ]
```

### With Mastra RAG

```typescript
import { createVectorQueryTool } from '@mastra/rag';
import { voyage } from '@mastra/voyageai';

const tool = createVectorQueryTool({
  vectorStore,
  model: voyage, // Embedder
  reranker: {
    model: voyage.reranker, // VoyageAI reranker
    options: { topK: 5 },
  },
});
```

### Pre-configured Reranker Models

```typescript
import { voyage } from '@mastra/voyageai';

// Default reranker (rerank-2.5)
voyage.reranker;

// Specific models
voyage.reranker25; // rerank-2.5
voyage.reranker25lite; // rerank-2.5-lite
voyage.reranker2; // rerank-2
voyage.reranker2lite; // rerank-2-lite

// Create custom
voyage.createReranker({ model: 'rerank-1', truncation: false });
```

## AI SDK Compatibility

The package exports models compatible with both AI SDK v5 (V2) and v6 (V3):

```typescript
// V3 (default, AI SDK v6)
const v3Model = voyageEmbedding('voyage-3.5');
v3Model.specificationVersion; // 'v3'

// V2 (AI SDK v5)
const v2Model = voyageEmbeddingV2('voyage-3.5');
v2Model.specificationVersion; // 'v2'

// Pre-configured V2 models
voyage.largeV2; // voyage-3-large with V2 interface
voyage.v35V2; // voyage-3.5 with V2 interface
```

## API Reference

### Types

```typescript
type VoyageTextModel =
  | 'voyage-4-large'
  | 'voyage-4'
  | 'voyage-4-lite'
  | 'voyage-4-nano'
  | 'voyage-code-4'
  | 'voyage-3-large'
  | 'voyage-3.5'
  | 'voyage-3.5-lite'
  | 'voyage-code-3'
  | 'voyage-finance-2'
  | 'voyage-law-2';

type VoyageMultimodalModel = 'voyage-multimodal-3' | 'voyage-multimodal-3.5';

type VoyageContextModel = 'voyage-context-3' | 'voyage-context-4';

type VoyageInputType = 'query' | 'document' | null;
type VoyageOutputDimension = 256 | 512 | 1024 | 2048;
type VoyageOutputDtype = 'float' | 'int8' | 'uint8' | 'binary' | 'ubinary';

type VoyageRerankerModel =
  'rerank-2.5' | 'rerank-2.5-lite' | 'rerank-2' | 'rerank-2-lite' | 'rerank-1' | 'rerank-lite-1';

interface VoyageRerankerConfig {
  model: VoyageRerankerModel;
  apiKey?: string;
  truncation?: boolean;
}
```

## License

Apache-2.0
