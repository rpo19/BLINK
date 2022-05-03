import express from 'express';
import api from '../api';
import { transformHTTPError } from '../utils/http-error';

export const expressLoader = () => {
  const app = express();

  app.use(express.urlencoded({ extended: true }))
  app.use(express.json())

  /**
   * All api endpoints are exposed under /api
   */
  app.use('/api', api());

  /**
   * Error handler
   */
  app.use((error, req, res, next) => {
    res.status(error.code).json({ ...transformHTTPError(error) })
  })

  console.log('Setup express... done');

  return app;
}