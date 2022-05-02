import mongoose from 'mongoose';

export const mongoLoader = () => {
    mongoose.connect(process.env.MONGO);
}