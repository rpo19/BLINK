import mongoose, { Schema } from 'mongoose';

const schema = new Schema({
  values: Object
});
export const Annotation = mongoose.model('Annotation', schema);

export const annotationDTO = (body) => {
  const _id = new mongoose.Types.ObjectId();
  return new Annotation({ _id, values: body.annotation });
}