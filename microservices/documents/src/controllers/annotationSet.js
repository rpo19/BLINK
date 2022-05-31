import { HTTP_ERROR_CODES, HTTPError } from '../utils/http-error';


export const AnnotationSetController = {
  insertOne: async (annotationSet) => {
    const doc = await annotationSet.save();
    return doc;
  }
}