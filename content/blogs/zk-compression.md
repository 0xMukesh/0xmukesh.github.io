---
title: Exploring ZK Compression
draft: false
date: 2024-04-07
---

If you've been active on Solana Twitter, you might have seen a lot of new exciting products being launched such as [blinks](https://solana.com/solutions/actions) and [ZK compression](https://www.zkcompression.com/). There is already a lot of developer content available for blinks and developers are already chewing glass and building blinks. Let's look at ZK compression - What the hell is it? What's the problem that ZK compression is trying to fix? How is it achieving it?

## What is ZK compression?

ZK compression is a new primitive built on Solana, by Helius Labs and Light Protocol, which enables developers to create applications at scale at cheaper costs via compressing on-chain data (which is stored in Solana's state), thereby reducing the cost of creating a new account on-chain by a lot.

"compressing on-chain data" doesn't mean that the data is initially being compressed via some lossy or lossless compression algorithm and is then decompressed later on. It might reduce the cost but not by a factor of 160x and 5000x.

| Creation Cost        | Regular Account | Compressed Account                |
| -------------------- | --------------- | --------------------------------- |
| 100-byte PDA Account | ~ 0.0016 SOL    | ~ 0.00001 SOL <br>(160x cheaper)  |
| 100 Token Accounts   | ~ 0.2 SOL       | ~ 0.00004 SOL <br>(5000x cheaper) |

ZK compression uses a similar approach as compressed NFTs, which leverages the idea of using state compression, which allows developers to store account data on less-pricey Solana's ledger while preserving the security, performance, and composability of the Solana L1.

In broad, ZK compression is mainly split into two parts - compression via concurrent merkle trees and zero-knowledge proofs to ensure the integrity of the compressed state.

### Concurrent Merkle Trees

In computer science, a Tree is a type of data structure that is a collection of objects (which are referred to as "nodes") that are linked to each other to represent a hierarchy. The top-most node is known as the "root node" and the bottom-most node is known as the "leaf node".

There are many sub-types of a tree, one of which is a binary tree in which there are two sub-trees similar to how are there two units of representation in binary system - 0 and 1

Merkle tree is a type of binary tree, in which every node which isn't a leaf node (referred to as "branch") is a cryptographic hash. The nodes are hashes of their children and the leaf nodes contain the data which is being hashed. Each branch is then also hashed together sequentially until only a single hash remains (referred to as "root hash").

After computing the root hash, data stored within a leaf node can be verified by rehashing the leaf's data along with hashes of adjacent branches. This is known as the proof path. Comparing this rehash to the root hash verifies the leaf data. If they match, the data is accurate. If not, the leaf data has been changed.

Whenever data in one of the leaf nodes is changed, the root hash is recomputed and the previous root hash becomes invalid, which is a major drawback of using a traditional Merkle tree during concurrent writes.

![](https://i.imgur.com/RS2QnCW.png)

Suppose there is a request to update $X_4$ to $X_4'$ (request `A`) and another concurrent request to update $X_6$ to $X_6''$ (request `B`). Assume that request `A` has been processed earlier with {$X_4'$, $X_5$, $X_3$} as proof. The root hash changes from $X_1$ to $X_1'$ after request `A` has been processed.

Due to this request `B` is invalidated, as the root hash has been changed, which can be mathematically expressed as:

$X_1' \neq H(H(X_6, X_7), X_2)$ (as $X_2$ has been changed to $X_2'$ due to request `A`)

If $X_2$ is replaced with $X_2'$ in request `B` then both the requests would be processed successfully. Here comes in concurrent Merkle tree, which keeps a record of past write updates performed on the Merkle tree. Light protocol has made their own implementation of [concurrent Merkle tree](https://github.com/Lightprotocol/light-protocol/tree/main/merkle-tree/concurrent), which stores change log and the current proof in an on-chain account for that corresponding Merkle tree which also stores final root hash.

In the above example,

1. Change log = $[X_4', X_2']$, it's the path of the tree which was changed by the last operation done on the tree
2. Current proof = $[X_7, X_2]$

The implementation finds the intersection node between the change log and current proof and replaces $X_2$ with $X_2'$ in the current proof. If you're curious to know what exactly happens under the hood, I'd recommend reading the "Concurrent Leaf Replacement" section in [Compressing Digital Assets with Concurrent Merkle Trees](https://drive.google.com/file/d/1BOpa5OFmara50fTvL0VIVYjtg-qzHCVc/view) whitepaper.

### Zero-knowledge proof

At the core, zero-knowledge proofs (ZKPs) allow one party (the prover) to prove to another (the verifier) that a statement is true, without revealing any information beyond the validity of the statement itself.

#### Where's Waldo?

Consider the following example, there are two people (Alice and Bob) who are trying to find Waldo in the below picture, which is from a children's book named [Where's Waldo](https://x.com/whereswaldo)

![](https://i.imgur.com/XkFpTPE.jpeg)

Alice has found Waldo but she doesn't want to reveal the exact location of Waldo to Bob she also wants to prove to Bob that she has found Waldo Alice being the prover, cuts a hole of the same size as Waldo in a very large, opaque sheet of cardboard. She places the picture under the cardboard such that only Waldo can be seen through the hole. She asks Bob to see through the hole - Bob can see Waldo but he doesn't know the exact coordinates of Waldo concerning the rest of the image as the cardboard is much bigger than the image.

This is an example of zero-knowledge proof where Alice proves to Bob that she knows where Waldo is but Bob doesn't gain any more knowledge except for the fact that Alice has found Waldo in the picture.

#### Two Balls

Consider the following example, Alice is colorblind but Bob is not colorblind. Alice doesn't believe Bob and Bob sets up an experiment that includes a _challenge and response_ mechanism, through which she can deduce whether Bob is lying or not.

Bob gives two balls of two different colors (one is red and another one is blue) to Alice to hold those two balls in her hands. Alice puts those balls behind her back and she can either exchange the balls between her hands or not. Afterwards, she will show those the balls to Bob and he'll respond with "yes" if the balls have been exchanged, or else, he'll respond with "no".

The experiment goes back and forth between Alice and Bob until Alice is convinced. In such an experiment, the probability of guessing the correct answer consecutively 20 times is 1 in 1,048,576. Such experiments, give a statistical guarantee that something could be true.

The "Where's Waldo?" example is an example of non-interactive zero-knowledge proof which didn't require any kind of challenge-response mechanism as in the "Two Balls" example, which is an example of interactive zero-knowledge proof.

ZK compression built by Helius labs and Light Protocol uses ZK-SNARK which stands for **Z**ero **K**nowledge **S**uccinct **N**on-interactive **AR**gument of **K**nowledge, which is known for its small proof size (128 bytes in case of ZK compression) and quick verification times. ZKPs is a vast and interesting topic in the field of mathematics and computer science which would be hard to summarize in this blog post, so here are a few resources related to ZKP, if you want to read more about ZK and ZK-SNARK:

1. [https://fisher.wharton.upenn.edu/wp-content/uploads/2020/09/Thesis_Terrence-Jo.pdf](https://fisher.wharton.upenn.edu/wp-content/uploads/2020/09/Thesis_Terrence-Jo.pdf)
2. [https://github.com/Lightprotocol/light-protocol/tree/main/circuit-lib](https://github.com/Lightprotocol/light-protocol/tree/main/circuit-lib)
3. [https://github.com/Lightprotocol/light-protocol/tree/main/light-prover](https://github.com/Lightprotocol/light-protocol/tree/main/light-prover)
4. [https://www.di.ens.fr/~nitulesc/files/Survey-SNARKs.pdf](https://www.di.ens.fr/~nitulesc/files/Survey-SNARKs.pdf)
5. [https://chriseth.github.io/notes/articles/zksnarks/zksnarks.pdf](https://chriseth.github.io/notes/articles/zksnarks/zksnarks.pdf)
6. [https://consensys.io/blog/introduction-to-zk-snarks](https://consensys.io/blog/introduction-to-zk-snarks) (written by team behind [`gnark`](https://github.com/Consensys/gnark), which is being used by [ZK compression](https://github.com/Lightprotocol/light-protocol/blob/main/light-prover/README.md))
7. [Zero Knowledge Proof (with Avi Wigderson) - Numberphile](https://www.youtube.com/watch?v=5ovdoxnfFVc)
8. [Computer Scientist Explains One Concept in 5 Levels of Difficulty](https://www.youtube.com/watch?v=fOGdb1CTu5c)

The good news is that you don't have to know about ZKPs in-depth to start developing applications using ZK compression.

## Developing apps using ZK compression

Enough of theory, let's jump into practicals and start building using ZK compression.

### Using Helius ZK compression RPC node

Let's use Helius Labs' ZK compression RPC node, instead of going through the hassle of setting up our local node. Let's write a simple typescript script that creates a token and mints some tokens into our `payer` account.

```ts
import stateless from "@lightprotocol/stateless.js"

const connection = stateless.createRpc(
  "https://zk-testnet.helius.dev:8899", // rpc
  "https://zk-testnet.helius.dev:8784", // zk compression rpc
  "https://zk-testnet.helius.dev:3001", // prover
)

const main = async () => {
  const health = await connection.getIndexerHealth()
  console.log(health)
}

main()
```

[Prover](https://github.com/Lightprotocol/light-protocol/blob/main/light-prover/README.md) is the service responsible for processing Merkle tree updates. The code is almost similar to how it is done using `@solana/web3.js`, except for the fact that we've had to add ZK compression RPC and prover while creating a connection object.

```ts
import { Rpc, confirmTx, createRpc } from "@lightprotocol/stateless.js"
import { createMint, mintTo } from "@lightprotocol/compressed-token"
import { Keypair } from "@solana/web3.js"

const payer = Keypair.generate()

const connection: Rpc = createRpc(
  "https://zk-testnet.helius.dev:8899", // rpc
  "https://zk-testnet.helius.dev:8784", // zk compression rpc
  "https://zk-testnet.helius.dev:3001", // prover
)

const main = async () => {
  console.log(`payer - ${payer.publicKey.toString()}`)

  await confirmTx(connection, await connection.requestAirdrop(payer.publicKey, 10e9))

  const { mint, transactionSignature } = await createMint(connection, payer, payer.publicKey, 6)

  console.log(`create-mint  success! txId: ${transactionSignature}`)

  const mintToTxId = await mintTo(connection, payer, mint, payer.publicKey, payer, 1e6)

  console.log(`mint-to success! txId: ${mintToTxId}`)
}

main()
```

In the above code snippet, we're creating a compressed token and minting 1 token into our `payer` wallet.

`createMint` function initializes a token mint account using [Solana Token Program](https://solana.fm/address/TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA) and creates a token pool PDA using [Light Protocol's Compressed Token Program](https://solana.fm/address/HXVfQ44ATEi9WBKLSCCwM54KokdkzqXci9xCQ7ST9SYN) ([`create_token_pool` source code](https://github.com/Lightprotocol/light-protocol/blob/a894188803d4753a528fd8656a5d5766a1381e15/programs/compressed-token/src/lib.rs#L37)).

Here's the signature of a transaction which is executed via `createMint` function - https://solana.fm/tx/49Pu29qgwJoEdLhE13rWUaRv5szFKHp966T58v41osP929Pa8rovn77hogeEQwrzW8FKxyiFFWbG9awA3pcj3gJX.

Make sure to use https://zk-testnet.helius.dev:8899 as the RPC URL in solana.fm

1. System Program allocates 86 bytes for [46ieAhhWgmLHjDbj5PjnWvK8knPxAhfVuYGH9y7Ud7A1](https://solana.fm/address/46ieAhhWgmLHjDbj5PjnWvK8knPxAhfVuYGH9y7Ud7A1) account
2. SPL Token Program initializes a mint account at the above address with mint authority as [AfiNiZ1M1UYeQWjCYmxMfoyGefLiigKmKc11zooHGPU9](https://solana.fm/address/AfiNiZ1M1UYeQWjCYmxMfoyGefLiigKmKc11zooHGPU9)
3. Light Protocol's Compressed Token Program creates token pool PDA ([4bqJdrQkUeQv2Bvm4T2fLpza48xqK3L4z3VCPnrUtQ5v](https://solana.fm/address/4bqJdrQkUeQv2Bvm4T2fLpza48xqK3L4z3VCPnrUtQ5v)) owned by SPL Token Program, with token mint as [46ieAhhWgmLHjDbj5PjnWvK8knPxAhfVuYGH9y7Ud7A1](https://solana.fm/address/46ieAhhWgmLHjDbj5PjnWvK8knPxAhfVuYGH9y7Ud7A1)

The `mintTo` function is responsible for minting the compressed tokens to the destination wallet (in this case, it's `payer`)

Here's the signature of a transaction that is executed via `mintTo` function - https://solana.fm/tx/283nNu7kSp7VVffUTqQT5LUQjw828K9R8dV3eAesch6unJEFaLZiYKTJfRYhDxL41iw1pSReb751txxT1jdJ644T

1. The mentioned amount of SPL token are minted to token pool PDA whose authority is the signer of the transaction ([`mint_spl_to_pool_pda` source code](https://github.com/Lightprotocol/light-protocol/blob/c3b7f49ab17fb7d8a0b0b68f809f735c27c38829/programs/compressed-token/src/process_mint.rs#L279))
2. The token data is serialized and the account data of every pubkey in the transaction is compressed ([`create_output_compressed_accounts` source code](https://github.com/Lightprotocol/light-protocol/blob/a894188803d4753a528fd8656a5d5766a1381e15/programs/compressed-token/src/process_transfer.rs#L166))
3. CPI is executed with the serialized data of `mintTo` transaction which is later parsed and stored in the database via indexers like [photon indexer](https://github.com/helius-labs/photon/) (built by Helius Labs) ([`cpi_execute_compressed_transaction_mint_to` source code](https://github.com/Lightprotocol/light-protocol/blob/c3b7f49ab17fb7d8a0b0b68f809f735c27c38829/programs/compressed-token/src/process_mint.rs#L137))

This was a quick example to get an idea of how ZK compression programs work. Other examples can be built using ZK compression, all the examples are available in [`examples`](https://github.com/Lightprotocol/light-protocol/tree/mains/examples) directory.

### Using local ZK compression node

For spinning up a local ZK compression node, you've to install ZK compression CLI and photon-indexer as well. Change `pnpm` with the package manager of your wish.

```bash
pnpm add @lightprotocol/zk-compression-cli -g
cargo install photon-indexer --version 0.30.0 --locked
```

After the installation step is done, run the following command which starts a single-node Solana cluster along with Photon RPC for indexing compression programs and Prover for processing Merkle trees.

```bash
light test-validator
```

## Limitations

ZK compression offers lower account rent costs but also has a few limitations, which must be kept in mind before opting to use ZK compression in your app:

1. Larger transaction size
2. High compute unit usage

### Larger transaction size

The maximum transaction size on Solana is 1232 bytes and if you read from at least 1 compressed account then 128 bytes are used up for validity proof, which is constant per transaction.

If you have a use case where an account's data is updated multiple times then its uncompressed version might be cheaper than the compressed version as every write operation has a small additional cost in contrast to the fixed per-byte cost for the uncompressed version. Whenever a transaction writes to a compressed account, it [nullifies](https://github.com/Lightprotocol/light-protocol/blob/a894188803d4753a528fd8656a5d5766a1381e15/programs/system/src/invoke/processor.rs#L205-L225) the previous compressed account state and appends the new compressed account as a leaf to the state tree, both of these actions add up to the additional cost.

Appending a compressed state to a state tree typically costs around 100-200 lamports per new leaf and nullifying a leaf in a state tree costs 5000 lamports per nullified leaf.

### High compute unit usage

A normal token transfer via the SPL Token program typically uses around 10k CU as compared to a compressed token transfer which uses around 292k CU.

Breakdown of usage of compute units in transactions that use ZK compression:

1. 100k CU for validity proof, it's constant per transaction if you read data from at least 1 compressed account
2. 100k CU used for [Poseidon](https://eprint.iacr.org/2019/458.pdf) hashing
3. 6k CU per compressed account read/write

Whenever Solana's global per-block CU limit (50 million CU) is reached, validator clients may prioritize transactions with higher per-CU priority fees, which would require your application's users to increase their priority fees.

## Resources

### ZKPs and zk-SNARKs

1. [Computer Scientist Explains One Concept in 5 Levels of Difficulty | WIRED](https://www.youtube.com/watch?v=fOGdb1CTu5c)
2. [Zero Knowledge Proof (with Avi Wigderson) - Numberphile](https://www.youtube.com/watch?v=5ovdoxnfFVc)
3. [Zero Knowledge Proofs - Computerphile](https://youtu.be/HUs1bH85X9I?si=KLC1h2OOsjOGH66L)
4. [Introduction to zk-SNARKs](https://consensys.io/blog/introduction-to-zk-snarks)
5. [An Exploration of Zero-Knowledge Proofs and zk-SNARKs](https://fisher.wharton.upenn.edu/wp-content/uploads/2020/09/Thesis_Terrence-Jo.pdf)
6. [zk-SNARKs in a Nutshell](https://chriseth.github.io/notes/articles/zksnarks/zksnarks.pdf)
7. [zk-SNARKs: A Gentle Introduction](https://www.di.ens.fr/~nitulesc/files/Survey-SNARKs.pdf)

### Developing using ZK compression

1. [Intro to Development](https://www.zkcompression.com/introduction/intro-to-development)
2. [ZK compression TypeScript Client](https://www.zkcompression.com/developers/typescript-client)
3. [ZK compression examples](https://github.com/Lightprotocol/light-protocol/tree/mains/examples)
