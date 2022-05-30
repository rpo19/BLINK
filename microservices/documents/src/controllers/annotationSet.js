import { HTTP_ERROR_CODES, HTTPError } from '../utils/http-error';


export const AnnotationSetController = {
  insertOne: async (annotationSet) => {
    try {
      const doc = await annotationSet.save();
      return doc;
    } catch (err) {
      throw new HTTPError({
        code: HTTP_ERROR_CODES.INTERNAL_SERVER_ERROR,
        message: 'Could not save annotation to DB.'
      })
    }
  }
}