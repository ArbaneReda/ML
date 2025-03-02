import React from "react";
import "./Home.css";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";

const Home = () => {
  return (
    <motion.div
      className="home-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      <div className="background-image"></div>

      <motion.h1
        className="home-title"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8 }}
      >
        MACHINE LEARNING & CYBERSÉCURITÉ
      </motion.h1>

      <p className="home-description">
        Ce site web représente notre travail sur le{" "}
        <span className="highlight">Machine Learning</span>. Il permet de tester
        et exécuter les <span className="highlight">scripts Python</span>{" "}
        directement depuis l'interface, rendant l'expérience plus fluide et
        interactive.
      </p>

      <p className="home-description">
        Nos travaux portent sur les{" "}
        <span className="highlight">algorithmes de Q-Learning</span> et de{" "}
        <span className="highlight">Deep Q-Networks</span>.
      </p>

      <div className="cards-container">
        <motion.div
          className="card"
          whileHover={{ scale: 1.05 }}
          transition={{ duration: 0.2 }}
        >
          <h2>Lab 0 - Q-Learning & Q-Network</h2>
          <p>
            Étude des <strong>Q-Tables</strong> et des{" "}
            <strong>Q-Networks</strong>. Nous avons exploré ces techniques pour
            optimiser l'apprentissage par renforcement.
            <a
              href="https://awjuliani.medium.com/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0"
              target="_blank"
              rel="noopener noreferrer"
              className="link"
            >
              Lire l'article
            </a>
          </p>
          <div className="button-container">
            <Link to="/lab0" className="button">
              Accéder au Lab 0
            </Link>
          </div>
        </motion.div>

        <motion.div
          className="card"
          whileHover={{ scale: 1.05 }}
          transition={{ duration: 0.2 }}
        >
          <h2>Lab 4 - Deep Q-Networks</h2>
          <p>
            Approfondissement des <strong>Deep Q-Networks (DQN)</strong> pour
            améliorer la prise de décision des IA.
            <a
              href="https://awjuliani.medium.com/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df"
              target="_blank"
              rel="noopener noreferrer"
              className="link"
            >
              Lire l'article
            </a>
          </p>
          <div className="button-container">
            <Link to="/lab4" className="button">
              Accéder au Lab 4
            </Link>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default Home;
