---
title: Deep dive into NFT compression
draft: false
date: 2023-04-14
tags: solana, web3
---

If you’re active on Solana Twitter, you might be familiar with the [DRiP](https://twitter.com/drip_haus) S2 drop. It took ~0.5 SOL for DRiP to airdrop ~100k “compressed” NFTs. Isn’t that nuts? If DRiP had to airdrop 100k uncompressed NFTs, it would cost them ~1.2k SOL. A whooping **2,400x difference** 🤯

NFT compression on Solana allows the users to mint a large number of NFTs for a fraction of the current cost.

**Heads up**: NFT compression doesn't mean that the "data" of the NFT is compressed into smaller bytes and stored which is decompressed later on. Well, this would decrease the cost of minting a NFT but not by a "huge" difference. The idea of "compressing and decompressing" the data of NFT won't scale at such a huge level where we could achieve a 2.4k difference.

![](https://i.imgur.com/WntBr0r.png)

# Data structures

## Tree

Tree is a type of data structure in computer science that is a collection of objects (known as “nodes”) that are linked to each other to represent a hierarchy.
The top-most node is known as the "root node" and the bottom-most nodes is known as the "leaf nodes".

![](https://i.imgur.com/tJNrI9W.png)

There are different types of tree data structures such as “Binary Tree”. The binary system consists of 1s and 0s, similarly binary tree consists of two sub-trees. There is a sub-type of binary tree known as “Binary search tree”, in which all the nodes on the left side are smaller than the nodes on the right side.

## Merkle tree

Merkle tree is a type of balanced binary tree i.e it has two sub-trees within it. In a Merkle tree, every node which isn’t a leaf (the inner nodes or the branches) are a cryptographic hash.

![](https://i.imgur.com/HDeTDWO.png)

In a Merkle tree, the nodes are hashes of its children and the leaf nodes contain the data which is being hashed. The children nodes are hashed using [SHA-256 algorithm](https://www.movable-type.co.uk/scripts/sha256.html).

Consider a Merkle tree of depth 2 and root node as $X_1$ and leaf nodes as $X_4$, $X_5$, $X_6$, $X_7$

```
	X1
      /    \
    X2      X3
   / \     / \
 X4  X5   X6  X7
```

The $X_2$ node is output of hashing $X_4$ and $X_5$ leaf nodes together and similarly for $X_3$ node.

- $X_2 = H(X_4, X_5)$
  $X_3 = H(X_6, X_7)$

The root node $X_1$ is the output of hashing the $X_2$ and $X_3$ nodes together

- $X_1 = H(H(X_4, X_5), X_3)$ and $\{X_4, X_5, X_3\}$ is known as the _"proof"_

If any of the leaf nodes is changed in a Merkle tree, the value of the root node changes completely. To check if two Merkle trees are similar, we would just need to check the equality of the root nodes.

# State and Ledger

Solana's state layer is an account database, a key-value store assigning account addresses to their metadata. The metadata includes balance of the account in [lamports](https://docs.solana.com/terminology#lamport), which program owns the account, whether the account is executable or not.

Using the state layer, we can find out how much SOL does an account hold but we can't find out _who_ sent the SOL to that specific account. This could be resolved by using ledger.

Ledger is a list of entries containing transactions signed by various other accounts. We can say that a transaction is a "state change" i.e the state can only change via a transaction.

# Account compression program

> Account compression program provides an interface for composing smart contracts to create and use SPL ConcurrentMerkleTrees. The primary application of using SPL ConcurrentMerkleTrees is to make edits to off-chain data with on-chain verification. [^1]

Native Solana tokens have multiple different accounts to store data on-chain such as token account, mint account, and metadata account.
Storing data on-chain costs SOL to maintain and it is funded by something called "rent". If an account maintains a minimum amount of SOL which is sufficient for 2 years of rent payment then the account is considered to be _"rent exempt"_. That's the reason why minting regular Solana NFT is quite expensive on scaling.

The account compression program creates an account for which stores the data in a Merkle tree and the root node's value (or _"root hash"_) is stored on-chain. Using account compression, a single Merkle tree can store data of multiple NFTs (specifically $2^{\text{depth}}$ as a Merkle tree has two sub-trees) making the NFT minting process cheap.

# Concurrent Merkle tree

Complications arise when considering concurrent writes to the same merkle tree. Every time a leaf node is modified, all the in-progress write request's proof would be invalid because the root of the merkle tree has been changed.

Suppose there is a request to update $X_4$ to $X_4'$ (request `A`) and another concurrent request to update $X_6$ to $X_6'$ (request `B`)

```
        X1'                         X1''
      /    \                      /     \
    X2'     X3                   X2     X3''
   / \     / \                  /  \   /  \
 X4'  X5  X6  X7              X4  X5  X6'' X7
   ----------                   -----------
    Request A                     Request B
```

Assume that request `A` got processed earlier with {$X_5$, $X_3$} as the proof. The entire path is updated and the root node changes to `X1'`.The request `B` would be then processed with {$X_7$, $X_2$} as the proof to the root node `X1`.
The request `B` would be invalidated as value of root node has been changed by request `A` earlier. Mathematically we can say that,

$X_1' \neq H(H(X_6, X_7), X_2)$

If there was a way to swap/replace `X2` with `X2'` in request `B` then both the request would succeed.

The account compression program keeps a record of past write updates performed on the merkle tree.
The store change log and the current proof can be represented in array form. It's known as "changelog buffer" and it's the path of tree which was changed by the last operation done on the tree.

Change log = $[X_4', X_2']$
Current proof = $[X_7, X_2]$

The program finds the intersection node amount the change log array and the current proof array and replaces $X_2$ with $X_2'$ to the current proof. If you're curious to know what happens under hood, I'd recommend to go through "Concurrent Leaf Replacement" section in the [whitepaper](https://file.notion.so/f/s/43c1516a-dc57-489d-afce-4e33ce44fbf9/Concurrent_Merkle_Tree_Whitepaper.pdf?id=c216e203-7cc8-4eb4-bc07-5e94f05dd415&table=block&spaceId=0f77f052-c0f3-4c86-aabc-4baed9a5006c&expirationTimestamp=1681224001094&signature=z4zK2jTS2Wov2qBb8r6AgTNpckWdi7op44QjdfK2D5A&downloadName=Concurrent+Merkle+Tree+Whitepaper.pdf)

# Bubblegum program

The Account Compression program (developed by Solana Labs) and Bubblegum program (developed by Metaplex Foundation) work in tandem to allow NFTs to be encoded into Solana ledger.

Take a look at the [this](https://solscan.io/tx/3TCzaubwLoSAUqoBjhDcfzuqHbvh57rWtxVrPTMVy5z6LYMsQZVZRYFaH3YxnD16jXG7XGt1obKCGhTpfUEvj3TV) transaction which mints a compressed NFT to an account

The transaction has a single instruction - "Bubblegum: Mint To Collection V1" which calls the Bubblegum program and mints a compressed NFT.
The transaction has 4 inner instructions:

- Bubblegum Set Collection Size
- Executes CPI to [`noop`](https://solscan.io/account/noopb9bkMVfRPU8AsbpTUg8AQkHtKwMYZiFUjNRtMmV) program
- Append data to the merkle tree
- Executes CPI to [`noop`](https://solscan.io/account/noopb9bkMVfRPU8AsbpTUg8AQkHtKwMYZiFUjNRtMmV) program

The "Set Collection Size" instruction changes the size of the [collection](https://solscan.io/token/DRiP2Pn2K6fuMLKQmt5rZWyHiUZ6WK3GChEySUpHSS4x) which "0387df" as the argument. "0387df" is a hexadecimal which translates to 231391.

The Bubblegum program executes a CPI to `noop` program where the data regarding the NFT is serialized into the instruction data. CPI calls are used instead of logging events to runtime as there is a 10kb log limit on Solana transactions.

The [Digital Asset Indexer](https://github.com/metaplex-foundation/digital-asset-rpc-infrastructure) listen to the Bubblegum transactions and parse the instruction data under the CPI of `noop` program and stores/updates them in an off-chain database. For example, when a compressed NFT is minted the indexer will parse the instruction data and extract the NFT information such as name, collection and owner's wallet address. If the merkle tree was not found in the off-chain database, the indexer will fetch the merkle tree's transaction history and reconstruct the state of the tree.

# Resources

- [Concurrent Merkle Tree Whitepaper](https://drive.google.com/file/d/1BOpa5OFmara50fTvL0VIVYjtg-qzHCVc/view)
- Account Compression Program
  - [Documentation](https://spl.solana.com/account-compression)
  - [Github](https://github.com/solana-labs/solana-program-library/tree/master/account-compression)
- [Bubblegum Program](https://github.com/metaplex-foundation/metaplex-program-library/tree/master/bubblegum/program)
- [Digital Asset Indexer](https://github.com/metaplex-foundation/digital-asset-rpc-infrastructure)
- Code examples
  - [metaplex-foundation/compression-read-api-js-examples](https://github.com/metaplex-foundation/compression-read-api-js-examples)
  - [solana-developers/compressed-nfts](https://github.com/solana-developers/compressed-nfts)

[^1]: https://spl.solana.com/account-compression
