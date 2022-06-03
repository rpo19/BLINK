import mongoose, { Schema } from 'mongoose';

// const annotationSchema = new Schema({
//   type: String,
//   start: Number,
//   end: Number,
//   id: Number,
//   features: Object
// })

const annotationSetSchema = new Schema({
  name: String, // always the same as the identifier ?
  // annotations: [annotationSchema],
  annotations: [Object],
  next_annid: Number
});
export const AnnotationSet = mongoose.model('AnnotationSet', annotationSetSchema);

export const annotationSetDTO = (annset) => {
  const _id = new mongoose.Types.ObjectId();
  annset._id = _id
  return new AnnotationSet(annset);
}