import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Header from "./components/Header/Header";
import Home from "./pages/Home/Home";
import Lab0 from "./pages/Lab0/Lab0";

function App() {
  return (
    <Router>
      <Header />
      <div className="app-container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/lab0" element={<Lab0 />} />
          <Route path="/lab4" element={<div>Lab 4 Content</div>} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
