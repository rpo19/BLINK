import { AnnotationSet } from '../models/annotationSet';
import { Document } from '../models/document';
import { HTTP_ERROR_CODES, HTTPError } from '../utils/http-error';


export const AnnotationSetController = {
  insertOne: async (annotationSet) => {
    const doc = await annotationSet.save();
    return doc;
  },
  deleteOne: async (docId, annotationSetId) => {
    const updatedDocument = await Document.updateOne({ id: docId }, {
      $pull: {
        annotation_sets: annotationSetId,
      },
    });
    // delete annotation set document
    return AnnotationSet.deleteOne({ _id: annotationSetId });
  }
}