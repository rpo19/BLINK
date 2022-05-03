import mongoose from 'mongoose';

const schema = new mongoose.Schema({
  title: String,
  preview: String,
  text: String,
  annotation: Object
});
export const Document = mongoose.model('Document', schema);

export const documentDTO = (body) => {
  const text = body.text;
  const preview = body.preview || body.text.slice(0, 400);
  const title = body.title || body.text.split(' ').slice(0, 3).join(' ');

  return new Document({ text, preview, title });
}