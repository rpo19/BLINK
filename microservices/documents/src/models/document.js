import mongoose from 'mongoose';

const schema = new mongoose.Schema({
    title: String,
    preview: String,
    text: String,
    annotation: Object
});
export const Document = mongoose.model('Document', schema);