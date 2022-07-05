import { Router } from 'express';
import { DocumentController } from '../controllers/document';
import { asyncRoute } from '../utils/async-route';
import { HTTPError, HTTP_ERROR_CODES } from '../utils/http-error';
import { documentDTO } from '../models/document';
import { validateRequest } from 'zod-express-middleware';
import { z } from 'zod';
import { annotationSetDTO } from '../models/annotationSet';
import { AnnotationSetController } from '../controllers/annotationSet';


const route = Router();

export default (app) => {
  // route base root
  app.use('/document', route);

  /**
   * Get all documents
   */
  route.get('/',
    validateRequest(
      {
        req: {
          query: z.object({
            // query to find by name
            q: z.string().optional(),
            // page
            page: z.number().optional(),
            // n. of documents to return for each page
            limit: z.number().optional(),
          })
        }
      }
    ), asyncRoute(async (req, res) => {
      const { q, limit, page } = req.query;
      const documentsPage = await DocumentController.findAll(q, limit, page);
      return res.json(documentsPage).status(200);
    }));

  /**
   * Get document by id
   */
  route.get('/:id', asyncRoute(async (req, res, next) => {
    const { id } = req.params;

    const document = await DocumentController.findOne(id);
    // convert annotation_sets from list to object
    var new_sets = {}
    for (const annset of document.annotation_sets) {
      // delete annset._id;

      // add mention to annotations features
      if (annset.name.startsWith('entities')) {
        for (const annot of annset.annotations) {
          if (!('mention' in annot.features)) {
            annot.features.mention = document.text.substring(annot.start, annot.end);
          }
        }
      }

      // ensure annset is sorted
      annset.annotations.sort((a, b) => a.start - b.start)

      new_sets[annset.name] = annset;
    }
    document.annotation_sets = new_sets;
    delete document.features.clusters;
    return res.json(document).status(200);
  }));

  /**
   * Create a new document
   */
  route.post('/',
    validateRequest(
      {
        req: {
          body: z.object({
            text: z.string(),
            annotation_sets: z.object(),
            preview: z.string().optional(),
            name: z.string().optional(),
            features: z.object().optional(),
            offset_type: z.string().optional()
          })
        }
      }
    ),
    asyncRoute(async (req, res, next) => {
      const annotationSets = []
      const annotationSetIds = []
      for (const [key, annset] of Object.entries(req.body.annotation_sets)) {
        var newAnnotationSet = annotationSetDTO(annset);
        annotationSetIds.push(newAnnotationSet._id);
        var newAnnotationSetDB = await AnnotationSetController.insertOne(newAnnotationSet);
        annotationSets.push(newAnnotationSetDB.toObject());
      }
      const newDocument = documentDTO(annotationSetIds, req.body);
      const doc = await DocumentController.insertOne(newDocument);

      return res.json({
        ...doc.toObject(),
        annotation_sets: annotationSets
      }).status(200)
    }));

  route.delete('/:docId/annotation-set/:annotationSetId',
    asyncRoute(async (req, res, next) => {
      const { docId, annotationSetId } = req.params;
      const result = AnnotationSetController.deleteOne(docId, annotationSetId);
      return res.json(result);
    }));
};