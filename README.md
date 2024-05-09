# Simple Smart Contract Project

## 📖 Overview

This project is a practical demonstration of using the Hardhat framework to develop, compile, deploy, and test Ethereum smart contracts written in Solidity. Inspired by the [Solidity, Blockchain, and Smart Contract Course – Beginner to Expert Python Tutorial](https://www.youtube.com/watch?v=gyMwXuJrbJQ&t=34221s) by Patrick Collins from freeCodeCamp, this project leverages Hardhat to run a local blockchain simulation, enabling thorough testing and debugging of smart contracts.

## ✨ Features

- **Solidity Development**: Smart contracts written in Solidity for Ethereum.
- **Local Blockchain and Sepolia testnet**: Utilization of Hardhat's local and testnet blockchain environment for development and testing.
- **Automated Testing**: Comprehensive test suites written in JavaScript to ensure contract functionality and security.

## 🚀 Getting Started

### 🔍 Prerequisites

- Node.js
- NPM (Node Package Manager) or yarn
- Hardhat

### 🛠 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/leonnloo/Simple-Smart-Contract
   ```
2. Navigate to the project directory (3 projects):
   ```bash
   cd <project-directory>
   ```
3. Install dependencies:
   ```bash
   npm install
   ```

### 📝 Compile Contracts

Compile the smart contracts using Hardhat:
```bash
npx hardhat compile
```

### 🌐 Deploy Contracts

Deploy the contracts to the local blockchain:
```bash
npx hardhat run scripts/deploy.js
```

### 🧪 Run Tests

Execute the test cases:
```bash
npx hardhat test
```

## 🤝 Contributing



Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, features, or improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙌 Acknowledgments

- Inspired by Patrick Collins and freeCodeCamp's comprehensive [Solidity and Blockchain course](https://www.youtube.com/watch?v=gyMwXuJrbJQ&t=34221s).
- Thanks to the Hardhat team for providing a robust framework for Ethereum development.
