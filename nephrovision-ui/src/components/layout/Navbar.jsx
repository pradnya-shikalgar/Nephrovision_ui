import React from 'react';
import { NavLink } from 'react-router-dom';
import { Activity, LayoutDashboard, Upload } from 'lucide-react';
import './Navbar.css';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Activity className="navbar-logo" />
        <span className="navbar-title">NephroVision</span>
      </div>
      <div className="navbar-links">
        <NavLink to="/" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>
          <LayoutDashboard size={18} /> Dashboard
        </NavLink>
        <NavLink to="/upload" className={({isActive}) => isActive ? 'nav-link active' : 'nav-link'}>
          <Upload size={18} /> Upload Image
        </NavLink>
      </div>
    </nav>
  );
};

export default Navbar;
