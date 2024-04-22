const { ethers, run, network } = require("hardhat")
// yarn hardhat run scripts/deploy.js --network sepolia
async function main() {
    const SimpleStorageFactory =
        await ethers.getContractFactory("SimpleStorage")
    // const SimpleStorageFactory = await ethers.deployContract("SimpleStorage")
    console.log("Deploying contract...")
    // console.log(SimpleStorageFactory)
    const simpleStorage = await SimpleStorageFactory.deploy()
    // await simpleStorage.deploymentTransaction(1)
    const deployTransaction = simpleStorage.deploymentTransaction(1)
    await deployTransaction.wait() // Wait for the transaction to be mined
    address = await simpleStorage.getAddress()
    console.log(`Deployed contract to: ${address}`)
    console.log(`TARGET Deployed contract to: ${simpleStorage.target}`)

    // console.log(network.config);
    // if (network.config.chainId === 11155111 && process.env.ETHERSCAN_API_KEY) {
    //     const deployTransaction = simpleStorage.deploymentTransaction(6)
    //     await deployTransaction.wait() // Wait for the transaction to be mined
    //     // await simpleStorage.deploymentTransaction(6)
    //     // await simpleStorage.wait()
    //     await verify(address, [])
    // }
    
    //Interact with contract
    const currentValue = await simpleStorage.retrieve()
    console.log(`Current Value is ${currentValue}`)
    
    //Update the current value
    const transactionResponse = await simpleStorage.store(7)
    await transactionResponse.wait(1)
    const updatedValue = await simpleStorage.retrieve()
    console.log(`Updated Value is ${updatedValue}`)
}

const verify = async (contractAddress, args) => {
    console.log("Verifying contract...")
    try {
        await run("verify:verify", {
            address: contractAddress,
            constructorArguments: args,
        })
    } catch (error) {
        if (error.message.toLowerCase().includes("already verified")){
            console.log("Already Verified!");
        }   else {
            console.log(error);
        }
    }
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error)
        process.exit(1)
    })
