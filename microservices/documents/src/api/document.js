import { Router } from 'express';
import { DocumentController } from '../controllers/document';
import { asyncRoute } from '../utils/async-route';
import { HTTPError, HTTP_ERROR_CODES } from '../utils/http-error';
import { Document } from '../models/document';


const route = Router();

export default (app) => {
  // route base root
  app.use('/document', route);

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
     route.put('/', asyncRoute(async (req, res, next) => {

      // find document by id
      const document = new Document();
      document.text = req.body.text;
      document.annotation = req.body.annotation;

      document.preview = req.body.preview || document.text.split(0,300);

      document.title = req.body.title || document.text.split(0,12);

      const output = await document.save();

      return res.json(output).status(200);
    }));
};