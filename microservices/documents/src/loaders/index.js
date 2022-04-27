import { expressLoader } from "./express";

export const startServer = (callback) => {
  const PORT = process.env.PORT;
  // setup express routes
  const app = expressLoader();
  // loader for database
  // ...

  // start server
  const server = app.listen(PORT, () => callback({ PORT }));

  return server;
}