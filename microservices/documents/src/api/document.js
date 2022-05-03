import { Router } from 'express';
import { DocumentController } from '../controllers/document';
import { asyncRoute } from '../utils/async-route';
import { HTTPError, HTTP_ERROR_CODES } from '../utils/http-error';
import { Document, documentDTO } from '../models/document';
import { validateRequest } from 'zod-express-middleware';
import { z } from 'zod';


const route = Router();

export default (app) => {
  // route base root
  app.use('/document', route);

  /**
   * Get all documents
   */
  route.get('/hello', asyncRoute(async (req, res) => {
    return res.json({ 'res': 'hello' }).status(200);
  }));

  /**
   * Get all documents
   */
  route.get('/', asyncRoute(async (req, res) => {
    // qui ci sarà la findAll di mongodb, guardare DocumentController per più info
    const documents = await DocumentController.findAll();
    return res.json(documents).status(200);
  }));

  /**
   * Get document by id
   */
  route.get('/:id', asyncRoute(async (req, res, next) => {
    const { id } = req.params;

    // find document by id
    const document = await DocumentController.findOne(id);

    if (!document) {
      throw new HTTPError({
        code: HTTP_ERROR_CODES.NOT_FOUND,
        message: `There is no document with id '${id}'`
      });
    }
    return res.json(document).status(200);
  }));

  /**
   * Get document by id
   */
  route.post('/', validateRequest({
    body: z.object({
      text: z.string(),
      preview: z.string().optional(),
      title: z.string().optional()
    }),
  }), asyncRoute(async (req, res, next) => {
    const newDocument = documentDTO(req.body);
    const doc = await newDocument.save();
    return res.json(doc).status(200);
  }));
};