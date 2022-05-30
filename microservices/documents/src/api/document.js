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
  route.get('/', asyncRoute(async (req, res) => {
    const documents = await DocumentController.findAll();
    return res.json(documents).status(200);
  }));

  /**
   * Get document by id
   */
  route.get('/:id', asyncRoute(async (req, res, next) => {
    const { id } = req.params;

    const document = await DocumentController.findOne(id);
    return res.json(document).status(200);
  }));

  /**
   * Create a new document
   */
  route.post('/', validateRequest({
    body: z.object({
      text: z.string(),
      annotation_sets: z.any().array(),
      preview: z.string().optional(),
      name: z.string().optional(),
      features: z.object().optional(),
      offset_type: z.string().optional()
    }),
  }), asyncRoute(async (req, res, next) => {
    const annotationSets = []
    const annotationSetIds = []
    for (const annset of req.body.annotation_sets) {
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
};