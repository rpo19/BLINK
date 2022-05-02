import { expressLoader } from "./express";
import { mongoLoader } from "./mongo";

export const startServer = (callback) => {
  const PORT = process.env.PORT;
  // setup express routes
  const app = expressLoader();
  // loader for database
  // ...

  mongoLoader();

  // start server
  const server = app.listen(PORT, () => callback({ PORT }));

  return server;
}