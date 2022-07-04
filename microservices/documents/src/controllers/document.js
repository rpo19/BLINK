
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
  findAll: async (limit = 20, cursor) => {
    let docs;
    if (!lastDocId) {
      // get first 5 docs
      docs = await Document.find().sort({ id: -1 }).limit(limit).lean();
    }
    else {
      // get next 5 docs according to that last document id
      docs = await Document.find({ id: { $lt: cursor } })
        .sort({ id: -1 }).limit(limit).lean()
    }
    return docs
    // try {
    //   const docs = await Document.find({}).lean();
    //   return docs;
    // } catch (err) {
    //   throw new HTTPError({
    //     code: HTTP_ERROR_CODES.INTERNAL_SERVER_ERROR,
    //     message: 'Could not read from DB.'
    //   })
    // }
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