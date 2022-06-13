import { Router } from 'express';
import { asyncRoute } from '../utils/async-route';
import { AnnotationSet } from '../models/annotationSet';
import { DocumentController } from '../controllers/document';



const route = Router();

export default (app) => {
  // route base root
  app.use('/save', route);

  /**
   * Save entity annotation set
   * // TODO: save of the whole document parts, for now this works
   */
  route.post('/', asyncRoute(async (req, res) => {
    const { entitiesAnnotations } = req.body;
    const id = entitiesAnnotations._id;
    const resUpdate = await DocumentController.updateEntitiesAnnotationSet(id, entitiesAnnotations);
    return res.json(resUpdate)
  }));
};