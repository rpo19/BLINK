import { Router } from 'express';
import { DocumentController } from '../controllers/document';
import { asyncRoute } from '../utils/async-route';
import { HTTPError, HTTP_ERROR_CODES } from '../utils/http-error';
import { documentDTO } from '../models/document';
import { validateRequest } from 'zod-express-middleware';
import { z } from 'zod';
import { annotationDTO } from '../models/annotation';
import { AnnotationController } from '../controllers/annotation';


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
      annotation: z.any().array(),
      preview: z.string().optional(),
      title: z.string().optional()
    }),
  }), asyncRoute(async (req, res, next) => {
    const newAnnotation = annotationDTO(req.body);
    const newDocument = documentDTO(newAnnotation._id, req.body);
    const annotation = await AnnotationController.insertOne(newAnnotation);
    const doc = await DocumentController.insertOne(newDocument);

    return res.json({
      ...doc.toObject(),
      annotation: annotation.toObject().annotation
    }).status(200)
  }));
};