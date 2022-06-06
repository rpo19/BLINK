
import { Document } from '../models/document';
import { HTTPError, HTTP_ERROR_CODES } from '../utils/http-error';



export const DocumentController = {
  insertOne: async (document) => {
    try {
      const doc = await document.save();
      return doc;
    } catch (err) {
      throw new HTTPError({
        code: HTTP_ERROR_CODES.INTERNAL_SERVER_ERROR,
        message: 'Could not save document to DB.'
      })
    }
  },
  findAll: async () => {
    try {
      const docs = await Document.find({}).lean();
      return docs;
    } catch (err) {
      throw new HTTPError({
        code: HTTP_ERROR_CODES.INTERNAL_SERVER_ERROR,
        message: 'Could not read from DB.'
      })
    }
  },
  findOne: async (id) => {

    const doc = await Document
      .findOne({ id: id })
      .populate({
        path: 'annotation_sets'
      })
      .lean()
      .exec()
    if (!doc) {
      throw new HTTPError({
        code: HTTP_ERROR_CODES.NOT_FOUND,
        message: `Document with id '${id}' was not found.`
      })
    }
    // return { ...doc, annotation_set: [...doc.annotation_sets.values] }
    return doc
  }
}