import { HTTP_ERROR_CODES, HTTPError } from '../utils/http-error';


export const AnnotationController = {
  insertOne: async (annotation) => {
    try {
      const doc = await annotation.save();
      return doc;
    } catch (err) {
      throw new HTTPError({
        code: HTTP_ERROR_CODES.INTERNAL_SERVER_ERROR,
        message: 'Could not save annotation to DB.'
      })
    }
  }
}