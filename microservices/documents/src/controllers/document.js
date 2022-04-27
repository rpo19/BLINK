import * as fs from 'fs/promises'
import { DOCUMENTS } from '../../_collections/documents';
import path from 'path';
import { AnnotationController } from './annotation';

export const DocumentController = {
  findAll: async () => {
    return Promise.all(Object.keys(DOCUMENTS).map(async (key) => {
      const { content: contentPath, annotations, ...rest } = DOCUMENTS[key];

      const documentPath = path.join(process.cwd(), '_files', contentPath);

      // penso che la preview viene salvata direttamente in un documento su mongo, non penso sia necessario
      // ricavarla tutte le volte che si fa una find
      const content = await fs.readFile(documentPath, 'utf-8');
      const preview = content.slice(0, 600);

      // quello che effettivamente dovrà essere ritornato
      const document = {
        ...rest,
        preview
      }

      return document;
    }));
  },
  findOne: async (id) => {
    const document = DOCUMENTS[id];
    if (!document) {
      return null;
    }
    const documentPath = path.join(process.cwd(), '_files', document.content);
    const content = await fs.readFile(documentPath, 'utf-8');
    // questa con mongo in realtà non esisterà più perchè le annotazioni sono direttamente
    // incluse nel documento. Anche se le annotazioni saranno in una diversa collection si
    // farà uso di populate (https://stackoverflow.com/questions/46967574/join-two-collections-using-mongoose-and-get-data-from-both)
    const annotations = await AnnotationController.readAnnotations(document.annotations);

    return {
      ...document,
      content,
      annotations
    }
  }
}