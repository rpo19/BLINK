
import { Document } from '../models/document';
import { AnnotationSet } from '../models/annotationSet';
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
  findAll: async (q = '', limit = 20, page = 1) => {

    const query = {
      ...(q && {
        name: { $regex: q, $options: 'i' }
      })
    }

    const options = {
      select: ['_id', 'id', 'name', 'preview'],
      page,
      limit
    };

    return Document.paginate(query, options);
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
    return doc
  },
  updateEntitiesAnnotationSet: async (id, annotationSet) => {
    try {
      const doc = await AnnotationSet.findByIdAndUpdate(id, {
        ...annotationSet
      })
      return { ok: true };
    } catch (err) {
      throw new HTTPError({
        code: HTTP_ERROR_CODES.INTERNAL_SERVER_ERROR,
        message: `Something went wrong when updating the Annotation Set.`
      })
    }
  }
}