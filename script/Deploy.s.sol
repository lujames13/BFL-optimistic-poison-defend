// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "forge-std/Script.sol";
import "../src/FederatedLearning.sol";
import "../src/libraries/KrumDefense.sol";

contract DeployFederatedLearning is Script {
    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        vm.startBroadcast(deployerPrivateKey);
        
        // Deploy FederatedLearning contract
        FederatedLearning flContract = new FederatedLearning();
        
        // Initialize the contract
        flContract.initialize();
        
        // Log deployed contract address
        console.log("FederatedLearning contract deployed at:", address(flContract));
        
        vm.stopBroadcast();
    }
}