import mongoose, { Schema } from 'mongoose';
import Inc from "mongoose-sequence";

const AutoIncrement = Inc(mongoose);

const schema = new mongoose.Schema({
  annotation_sets: [
    { type: Schema.Types.ObjectId, ref: 'AnnotationSet' }
  ],
  name: String,
  preview: String,
  text: String,
  features: Object,
  offset_type: String, // "p" for python style
});

schema.plugin(AutoIncrement, { inc_field: 'id' });
export const Document = mongoose.model('Document', schema);

export const documentDTO = (annotationSetIds, body) => {
  const text = body.text;
  const preview = body.preview || body.text.slice(0, 400);
  const name = body.name || body.text.split(' ').slice(0, 3).join(' ');
  const features = body.features;
  const offset_type = body.offset_type || "p";
  return new Document({ annotation_sets: annotationSetIds, name, preview, text, features, offset_type });
}