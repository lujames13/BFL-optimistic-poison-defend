// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

import "forge-std/Test.sol";

contract SetupTest is Test {
    function setUp() public {}

    function testEnvironmentSetup() public {
        assertTrue(true, "Foundry is properly set up");
    }
}
