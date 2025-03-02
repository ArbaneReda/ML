import React from "react";
import { Link } from "react-router-dom";
import "./Header.css";

const Header = () => {
  return (
    <nav className="header-container">
      <h1 className="header-title">
        🚀 Machine Learning - Master 1 Cybersécurité UPC
      </h1>
      <div className="nav-links">
        <Link to="/lab0" className="nav-item">
          Lab 0
        </Link>
        <Link to="/lab4" className="nav-item">
          Lab 4
        </Link>
      </div>
    </nav>
  );
};

export default Header;
