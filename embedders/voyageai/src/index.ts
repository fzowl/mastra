/**
 * @mastra/voyageai - VoyageAI Embeddings Integration for Mastra
 *
 * Provides text, multimodal, and contextualized chunk embeddings using the official VoyageAI SDK.
 *
 * @example Text Embeddings
 * ```typescript
 * import { voyage, voyageEmbedding } from '@mastra/voyageai';
 *
 * // Use default model (voyage-3.5)
 * const result = await voyage.doEmbed({ values: ['Hello world'] });
 *
 * // Or use specific model with options
 * const model = voyageEmbedding({
 *   model: 'voyage-3-large',
 *   inputType: 'query',
 *   outputDimension: 512,
 * });
 * ```
 *
 * @example With Mastra Memory
 * ```typescript
 * import { Memory } from '@mastra/memory';
 * import { PgVector } from '@mastra/pg';
 * import { voyage } from '@mastra/voyageai';
 *
 * const memory = new Memory({
 *   vector: new PgVector(connectionString),
 *   embedder: voyage,
 *   options: { semanticRecall: { topK: 5 } },
 * });
 * ```
 *
 * @example Multimodal Embeddings
 * ```typescript
 * import { voyageMultimodalEmbedding } from '@mastra/voyageai';
 *
 * const multimodal = voyageMultimodalEmbedding('voyage-multimodal-3.5');
 * const result = await multimodal.doEmbed({
 *   values: [{
 *     content: [
 *       { type: 'text', text: 'A cat playing' },
 *       { type: 'image_url', image_url: 'https://example.com/cat.jpg' }
 *     ]
 *   }]
 * });
 * ```
 *
 * @example Contextualized Chunk Embeddings
 * ```typescript
 * import { voyageContextualizedEmbedding } from '@mastra/voyageai';
 *
 * // Each input string is embedded independently as its own document.
 * const contextual = voyageContextualizedEmbedding('voyage-context-4');
 * const result = await contextual.doEmbed({
 *   values: ['First document...', 'Second document...'],
 * });
 *
 * // Embed a query for retrieval
 * const query = await contextual.doEmbed({
 *   values: ['What was the revenue?'],
 *   providerOptions: { voyage: { inputType: 'query' } },
 * });
 * ```
 */

// Re-export all types
export * from './types';

export { VoyageAIClient } from 'voyageai';
export type { TokenizeResult } from 'voyageai';

// Re-export embedding model classes
export {
  VoyageTextEmbeddingModelV2,
  VoyageTextEmbeddingModelV3,
  createVoyageTextEmbedding,
  createVoyageTextEmbeddingV2,
} from './text-embedding';

export { VoyageMultimodalEmbeddingModel, createVoyageMultimodalEmbedding } from './multimodal-embedding';

export {
  VoyageContextualizedEmbeddingModelV2,
  VoyageContextualizedEmbeddingModelV3,
  createVoyageContextualizedEmbedding,
  createVoyageContextualizedEmbeddingV2,
} from './contextualized-embedding';

export { VoyageRelevanceScorer, createVoyageReranker, voyageReranker, type RelevanceScoreProvider } from './reranker';

// Import for convenience object
import {
  createVoyageTextEmbedding,
  createVoyageTextEmbeddingV2,
  VoyageTextEmbeddingModelV3,
  VoyageTextEmbeddingModelV2,
} from './text-embedding';
import { createVoyageMultimodalEmbedding, VoyageMultimodalEmbeddingModel } from './multimodal-embedding';
import {
  createVoyageContextualizedEmbedding,
  createVoyageContextualizedEmbeddingV2,
  VoyageContextualizedEmbeddingModelV3,
} from './contextualized-embedding';
import { createVoyageReranker, VoyageRelevanceScorer } from './reranker';

// ============================================================================
// Convenience Factory Functions
// ============================================================================

/**
 * Create a VoyageAI text embedding model (V3)
 *
 * @param config - Model name or full configuration
 * @returns EmbeddingModelV3 compatible model
 */
export const voyageEmbedding = createVoyageTextEmbedding;

/**
 * Create a VoyageAI text embedding model (V2)
 *
 * @param config - Model name or full configuration
 * @returns EmbeddingModelV2 compatible model
 */
export const voyageEmbeddingV2 = createVoyageTextEmbeddingV2;

/**
 * Create a VoyageAI multimodal embedding model
 *
 * @param config - Model name or full configuration
 * @returns VoyageMultimodalEmbeddingModel instance
 */
export const voyageMultimodalEmbedding = createVoyageMultimodalEmbedding;

/**
 * Create a VoyageAI contextualized chunk embedding model (V3)
 *
 * @param config - Model name or full configuration
 * @returns EmbeddingModelV3 compatible model
 */
export const voyageContextualizedEmbedding = createVoyageContextualizedEmbedding;

/**
 * Create a VoyageAI contextualized chunk embedding model (V2)
 *
 * @param config - Model name or full configuration
 * @returns EmbeddingModelV2 compatible model
 */
export const voyageContextualizedEmbeddingV2 = createVoyageContextualizedEmbeddingV2;

// ============================================================================
// Convenience Object with Pre-configured Models
// ============================================================================

/**
 * Pre-configured VoyageAI embedding models
 *
 * Default export provides the voyage-3.5 model as the default.
 * Access specific models through named properties.
 *
 * @example
 * ```typescript
 * import { voyage } from '@mastra/voyageai';
 *
 * // Default model (voyage-3.5)
 * const result = await voyage.doEmbed({ values: ['Hello'] });
 *
 * // Specific models
 * const largeResult = await voyage.large.doEmbed({ values: ['Hello'] });
 * const codeResult = await voyage.code.doEmbed({ values: ['function foo() {}'] });
 *
 * // Multimodal
 * const multimodalResult = await voyage.multimodal.doEmbed({
 *   values: [{ content: [{ type: 'text', text: 'Hello' }] }]
 * });
 *
 * // Contextualized (each input embedded independently)
 * const contextResult = await voyage.contextualized.doEmbed({
 *   values: ['document one', 'document two'],
 * });
 * ```
 */
export const voyage: VoyageTextEmbeddingModelV3 & {
  // Text models (V3) - voyage-4 series
  v4large: VoyageTextEmbeddingModelV3;
  v4: VoyageTextEmbeddingModelV3;
  v4lite: VoyageTextEmbeddingModelV3;
  v4nano: VoyageTextEmbeddingModelV3;
  code4: VoyageTextEmbeddingModelV3;

  // Text models (V3) - voyage-3 series
  large: VoyageTextEmbeddingModelV3;
  v35: VoyageTextEmbeddingModelV3;
  v35lite: VoyageTextEmbeddingModelV3;
  code: VoyageTextEmbeddingModelV3;
  finance: VoyageTextEmbeddingModelV3;
  law: VoyageTextEmbeddingModelV3;

  // Text models (V2 for backward compatibility) - voyage-4 series
  v4largeV2: VoyageTextEmbeddingModelV2;
  v4V2: VoyageTextEmbeddingModelV2;
  v4liteV2: VoyageTextEmbeddingModelV2;
  v4nanoV2: VoyageTextEmbeddingModelV2;
  code4V2: VoyageTextEmbeddingModelV2;

  // Text models (V2 for backward compatibility) - voyage-3 series
  largeV2: VoyageTextEmbeddingModelV2;
  v35V2: VoyageTextEmbeddingModelV2;
  v35liteV2: VoyageTextEmbeddingModelV2;
  codeV2: VoyageTextEmbeddingModelV2;
  financeV2: VoyageTextEmbeddingModelV2;
  lawV2: VoyageTextEmbeddingModelV2;

  // Multimodal models
  multimodal: VoyageMultimodalEmbeddingModel;
  multimodal3: VoyageMultimodalEmbeddingModel;
  multimodal35: VoyageMultimodalEmbeddingModel;

  // Contextualized model
  contextualized: VoyageContextualizedEmbeddingModelV3;
  context3: VoyageContextualizedEmbeddingModelV3;
  context4: VoyageContextualizedEmbeddingModelV3;

  // Reranker models
  reranker: VoyageRelevanceScorer;
  reranker25: VoyageRelevanceScorer;
  reranker25lite: VoyageRelevanceScorer;
  reranker2: VoyageRelevanceScorer;
  reranker2lite: VoyageRelevanceScorer;

  // Factory functions
  embedding: typeof createVoyageTextEmbedding;
  embeddingV2: typeof createVoyageTextEmbeddingV2;
  multimodalEmbedding: typeof createVoyageMultimodalEmbedding;
  contextualizedEmbedding: typeof createVoyageContextualizedEmbedding;
  createReranker: typeof createVoyageReranker;
} = (() => {
  // Lazy cache to avoid import-time crashes when VOYAGE_API_KEY is not set
  const cache = new Map<string, any>();
  function lazy<T>(key: string, factory: () => T): T {
    let instance = cache.get(key);
    if (!instance) {
      instance = factory();
      cache.set(key, instance);
    }
    return instance as T;
  }

  // The base object delegates doEmbed to a lazily-created default model
  const base = {
    get specificationVersion() {
      return 'v3' as const;
    },
    get provider() {
      return 'voyage' as const;
    },
    get modelId() {
      return 'voyage-3.5';
    },
    get maxEmbeddingsPerCall() {
      return 1000;
    },
    get supportsParallelCalls() {
      return true;
    },
    doEmbed(args: any) {
      return lazy('_default', () => createVoyageTextEmbedding('voyage-3.5')).doEmbed(args);
    },
  };

  return Object.defineProperties(base, {
    // Text models (V3) - voyage-4 series
    v4large: { get: () => lazy('v4large', () => createVoyageTextEmbedding('voyage-4-large')) },
    v4: { get: () => lazy('v4', () => createVoyageTextEmbedding('voyage-4')) },
    v4lite: { get: () => lazy('v4lite', () => createVoyageTextEmbedding('voyage-4-lite')) },
    v4nano: { get: () => lazy('v4nano', () => createVoyageTextEmbedding('voyage-4-nano')) },
    code4: { get: () => lazy('code4', () => createVoyageTextEmbedding('voyage-code-4')) },
    // Text models (V3) - voyage-3 series
    large: { get: () => lazy('large', () => createVoyageTextEmbedding('voyage-3-large')) },
    v35: { get: () => lazy('v35', () => createVoyageTextEmbedding('voyage-3.5')) },
    v35lite: { get: () => lazy('v35lite', () => createVoyageTextEmbedding('voyage-3.5-lite')) },
    code: { get: () => lazy('code', () => createVoyageTextEmbedding('voyage-code-3')) },
    finance: { get: () => lazy('finance', () => createVoyageTextEmbedding('voyage-finance-2')) },
    law: { get: () => lazy('law', () => createVoyageTextEmbedding('voyage-law-2')) },
    // Text models (V2) - voyage-4 series
    v4largeV2: { get: () => lazy('v4largeV2', () => createVoyageTextEmbeddingV2('voyage-4-large')) },
    v4V2: { get: () => lazy('v4V2', () => createVoyageTextEmbeddingV2('voyage-4')) },
    v4liteV2: { get: () => lazy('v4liteV2', () => createVoyageTextEmbeddingV2('voyage-4-lite')) },
    v4nanoV2: { get: () => lazy('v4nanoV2', () => createVoyageTextEmbeddingV2('voyage-4-nano')) },
    code4V2: { get: () => lazy('code4V2', () => createVoyageTextEmbeddingV2('voyage-code-4')) },
    // Text models (V2) - voyage-3 series
    largeV2: { get: () => lazy('largeV2', () => createVoyageTextEmbeddingV2('voyage-3-large')) },
    v35V2: { get: () => lazy('v35V2', () => createVoyageTextEmbeddingV2('voyage-3.5')) },
    v35liteV2: { get: () => lazy('v35liteV2', () => createVoyageTextEmbeddingV2('voyage-3.5-lite')) },
    codeV2: { get: () => lazy('codeV2', () => createVoyageTextEmbeddingV2('voyage-code-3')) },
    financeV2: { get: () => lazy('financeV2', () => createVoyageTextEmbeddingV2('voyage-finance-2')) },
    lawV2: { get: () => lazy('lawV2', () => createVoyageTextEmbeddingV2('voyage-law-2')) },
    // Multimodal models
    multimodal: { get: () => lazy('multimodal', () => createVoyageMultimodalEmbedding('voyage-multimodal-3.5')) },
    multimodal3: { get: () => lazy('multimodal3', () => createVoyageMultimodalEmbedding('voyage-multimodal-3')) },
    multimodal35: { get: () => lazy('multimodal35', () => createVoyageMultimodalEmbedding('voyage-multimodal-3.5')) },
    // Contextualized model
    contextualized: {
      get: () => lazy('contextualized', () => createVoyageContextualizedEmbedding('voyage-context-3')),
    },
    context3: { get: () => lazy('context3', () => createVoyageContextualizedEmbedding('voyage-context-3')) },
    context4: { get: () => lazy('context4', () => createVoyageContextualizedEmbedding('voyage-context-4')) },
    // Reranker models
    reranker: { get: () => lazy('reranker', () => createVoyageReranker('rerank-2.5')) },
    reranker25: { get: () => lazy('reranker25', () => createVoyageReranker('rerank-2.5')) },
    reranker25lite: { get: () => lazy('reranker25lite', () => createVoyageReranker('rerank-2.5-lite')) },
    reranker2: { get: () => lazy('reranker2', () => createVoyageReranker('rerank-2')) },
    reranker2lite: { get: () => lazy('reranker2lite', () => createVoyageReranker('rerank-2-lite')) },
    // Factory functions (no lazy needed - they are factories themselves)
    embedding: { value: createVoyageTextEmbedding },
    embeddingV2: { value: createVoyageTextEmbeddingV2 },
    multimodalEmbedding: { value: createVoyageMultimodalEmbedding },
    contextualizedEmbedding: { value: createVoyageContextualizedEmbedding },
    createReranker: { value: createVoyageReranker },
  }) as any;
})();

// Default export
export default voyage;
