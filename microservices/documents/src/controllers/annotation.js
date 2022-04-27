import path from 'path';
import * as fs from 'fs/promises';


export const AnnotationController = {
  readAnnotations: async (annotationsPath) => {
    const documentAnnotationPath = path.join(process.cwd(), '_files', annotationsPath);
    const annotations = JSON.parse(await fs.readFile(documentAnnotationPath, 'utf8'));

    // la trasformazione non sarà necessaria perchè saranno direttamente salvati così
    const processedAnnotations = annotations.map((annotation) => {
      const isLinked = annotation.candidates && annotation.candidates.length > 0;

      let topCandidate = {};
      let candidates = [];
      if (isLinked) {
        const [top, ...rest] = annotation.candidates;
        topCandidate = {
          // id of the entity inside entityStore (optional atm)
          // bisogna capire se ci sarà questo entity store
          id: 'entityStoreId',
          // per generalizzare (supponendo che non ci sia solo wikipedia come KB)
          baseUri: 'https://it.wikipedia.org/wiki?curid=',
          resourceId: annotation.top_wikipedia_id,
          score: top.score,
          normScore: top.norm_score,
          title: top.title,
          abstract: "Qui ci sarà l'abstract recuperato dalla KB",
          // optional
          type: ['/type/entity/KB', '/type/entity/KB'],
          // optional
          img: "/url/immagine/KB",
        }
        candidates = rest;
      }


      return {
        mention: annotation.mention,
        start: annotation.start_pos_original,
        end: annotation.end_pos_original,
        type: annotation.ner_type,
        // ner confidence score (optional atm)
        confidence: 1,
        topCandidate,
        candidates: candidates.map((candidate) => ({
          // id of the entity inside entityStore (optional atm)
          // bisogna capire se ci sarà questo entity store
          id: 'entityStoreId',
          baseUri: 'https://it.wikipedia.org/wiki?curid=',
          resourceId: candidate.wikipedia_id,
          title: candidate.title,
          score: candidate.score,
          normScore: candidate.norm_score
        }))
      }
    })

    return processedAnnotations;
  }
}