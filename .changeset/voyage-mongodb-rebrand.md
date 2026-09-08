---
'@mastra/voyageai': minor
---

Add the `voyage-code-4` text embedding model and accept both input formats for contextualized chunk embeddings.

- `voyage-code-4` is available through `voyageEmbedding('voyage-code-4')`, the `voyage.code4` / `voyage.code4V2` presets, and the `VoyageTextModel` type.
- `VoyageContextualizedEmbeddingModel.doEmbed` and `doEmbedGrouped` now accept `values: string[][] | string[]`, matching the official `contextualized_embed` API which supports `inputs: Union[List[List[str]], List[str]]`. See https://docs.voyageai.com/docs/contextualized-chunk-embeddings.

Documentation now refers to the provider as "VoyageAI by MongoDB".
