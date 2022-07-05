
import { Document } from '../models/document';
import { AnnotationSet } from '../models/annotationSet';
import { HTTPError, HTTP_ERROR_CODES } from '../utils/http-error';
import { annotationSetDTO } from '../models/annotationSet';
import { AnnotationSetController } from './annotationSet';

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
  updateEntitiesAnnotationSet: async (docId, annotationSets) => {
    const update = async (annotationSet) => {
      if (annotationSet._id) {
        console.log('Updating existing annotation set');
        return AnnotationSet.findByIdAndUpdate(annotationSet._id, {
          ...annotationSet
        })
      }
      console.log('Creating new annotation set');
      const newAnnotationSet = annotationSetDTO(annotationSet);
      // update document with a new annotation set
      const updatedDoc = await Document.findByIdAndUpdate(docId,
        { $push: { annotation_sets: newAnnotationSet._id } }
      )
      // add new annotation set
      return AnnotationSetController.insertOne(newAnnotationSet);
    }
    const updaters = Object.values(annotationSets).map((set) => update(set));
    return Promise.all(updaters);
  }
}