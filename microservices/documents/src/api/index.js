import { Router } from 'express';
import document from './document';

/**
 * Export all defined routes
 */
export default () => {
  const app = Router();
  document(app);

  return app
}