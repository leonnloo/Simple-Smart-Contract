require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config()

// require("@nomiclabs/hardhat-etherscan")
require("./tasks/block-number")

const SEPOLIA_RPC_URL = process.env.SEPOLIA_RPC_URL
const PRIVATE_KEY = process.env.PRIVATE_KEY
const ETHERSCAN_API_KEY = process.env.ETHERSCAN_API_KEY
/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  defaultNetwork: "hardhat",
  "networks": {
    sepolia: {
      url: SEPOLIA_RPC_URL,
      accounts: [PRIVATE_KEY],
      chainId: 11155111
    },
    localhost: {
      url: "http://localhost:8545",
      chainId: 31337,
    },
  },
  solidity: "0.8.24",
  etherscan: {
    apiKey: {
      sepolia: ETHERSCAN_API_KEY,
    }
  },
  sourcify: {
    enabled: true
  }
};
