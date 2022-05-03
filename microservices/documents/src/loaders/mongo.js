import mongoose from 'mongoose';

export const mongoLoader = async () => {
  try {
    await mongoose.connect(process.env.MONGO);
    console.log('Setup mongodb... done');
  } catch (err) {
    console.log(err)
  }
}