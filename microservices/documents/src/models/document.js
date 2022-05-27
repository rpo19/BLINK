import mongoose, { Schema } from 'mongoose';
import Inc from "mongoose-sequence";

const AutoIncrement = Inc(mongoose);

const schema = new mongoose.Schema({
  title: String,
  preview: String,
  text: String,
  annotation: { type: Schema.Types.ObjectId, ref: 'Annotation' }
});

schema.plugin(AutoIncrement, { inc_field: 'id' });
export const Document = mongoose.model('Document', schema);

export const documentDTO = (annotationId, body) => {
  const text = body.text;
  const preview = body.preview || body.text.slice(0, 400);
  const title = body.title || body.text.split(' ').slice(0, 3).join(' ');
  return new Document({ text, preview, title, annotation: annotationId });
}