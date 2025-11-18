# NOP Tiling Constraints - Generic

> **File:** `Deeploy/Targets/Generic/TileConstraints/NOPTileConstraint.py`
> **Class:** `NOPTileConstraint`
> **Last Updated:** 2025-11-18

---

## Table of Contents

1. [Overview](#overview)
2. [Tensor Definitions](#tensor-definitions)
3. [Geometrical Constraints](#geometrical-constraints)
4. [Policy Constraints](#policy-constraints)
5. [Buffer Management](#buffer-management)
6. [Examples](#examples)
7. [Notes and Limitations](#notes-and-limitations)

---

## Overview

### Description

The `NOPTileConstraint` (No-Operation Tile Constraint) is a minimal constraint class used for operators that pass through data without transformation or for placeholder operations. It sets up dimension variables without imposing strict equality constraints between input and output, allowing the tiler maximum flexibility. Additionally, it manages buffer deployment by marking unused auxiliary tensors as non-deployable.

### Supported Operations

- Identity/pass-through operations
- Reshape operations (when shapes are compatible)
- View operations
- Operators with auxiliary buffers that don't need deployment

### Parent Class

- **Inherits from:** `TileConstraint`

---

## Tensor Definitions

### Input Tensors

| Name | Shape | Description |
|------|-------|-------------|
| `data_in` | `[D_0, D_1, ..., D_n]` | Primary input tensor |

### Output Tensors

| Name | Shape | Description |
|------|-------|-------------|
| `data_out` | `[E_0, E_1, ..., E_m]` | Primary output tensor |

### Auxiliary Tensors

Any additional tensors in `parseDict` that are registered in the context (either global or local) are identified and marked as non-deployable if they are not the primary input or output.

### Parse Dictionary Keys

| Key | Type | Description |
|-----|------|-------------|
| `data_in` | `str` | Name of input buffer in context |
| `data_out` | `str` | Name of output buffer in context |
| (other keys) | `str` | Potential auxiliary buffer names |

---

## Geometrical Constraints

The `NOPTileConstraint` imposes minimal geometrical constraints, providing bounds rather than equalities.

### Dimension Variables

For each buffer (input and output), dimension variables are created:

| Variable | Tensor | Description |
|----------|--------|-------------|
| `dimVar[i]` | `data_in` or `data_out` | Dimension i of the tensor |

### Constraints

#### Upper Bound Constraint

**Proposition:**
```
For each buffer in [data_in, data_out]:
    For all i in [0, N-1]:
        dimVar[i] <= buffer.shape[i]
```

**Description:** Each dimension variable is constrained to be less than or equal to the full tensor dimension. This allows tiling (smaller dimensions) but prevents exceeding the actual tensor size.

**Note:** Unlike other tile constraints, there are no explicit equality constraints between input and output dimensions. This makes `NOPTileConstraint` suitable for operations where input and output may have different shapes.

---

## Policy Constraints

**None.** The `NOPTileConstraint` does not impose any policy constraints. The tiler has maximum flexibility in choosing tile sizes.

---

## Buffer Management

### Auxiliary Buffer Handling

The constraint identifies all string values in `parseDict` that reference buffers in the network context (either global or local). Buffers that are not the primary input or output are marked as non-deployable:

```python
for key, value in parseDict.items():
    if isinstance(value, str) and (ctxt.is_global(value) or ctxt.is_local(value)):
        pointer.append(value)

for bufferName in pointer:
    if bufferName not in [inputBufferName, outputBufferName]:
        ctxt.lookup(bufferName)._deploy = False
```

**Purpose:** This prevents auxiliary buffers (like shape tensors or metadata) from being allocated in the generated code when they are only used for compile-time information.

---

## Tiling Schedule

The `NOPTileConstraint` does not override `serializeTilingSolution`, so it uses the default implementation from the parent `TileConstraint` class. This means:

- Input and output cubes are typically treated as having the same structure
- Standard base address extraction is performed
- No custom replacements are generated

---

## Examples

### Example 1: Identity Operation

**Input Parameters:**
- Input shape: `[1, 32, 32, 64]`
- Output shape: `[1, 32, 32, 64]` (same as input)

**Constraints Generated:**
```
inputDimVar[0] <= 1
inputDimVar[1] <= 32
inputDimVar[2] <= 32
inputDimVar[3] <= 64

outputDimVar[0] <= 1
outputDimVar[1] <= 32
outputDimVar[2] <= 32
outputDimVar[3] <= 64
```

**Tiler Freedom:** The tiler can choose any tile size up to the full tensor dimensions.

---

### Example 2: Reshape with Auxiliary Shape Tensor

**Parse Dictionary:**
```python
{
    'data_in': 'input_tensor',
    'data_out': 'output_tensor',
    'shape': 'shape_tensor'  # Auxiliary tensor with new shape
}
```

**Buffer Management:**
- `input_tensor`: Deployed (primary input)
- `output_tensor`: Deployed (primary output)
- `shape_tensor`: Marked as `_deploy = False` (auxiliary, not needed at runtime)

---

### Example 3: Pass-through with Multiple Auxiliary Buffers

**Parse Dictionary:**
```python
{
    'data_in': 'features',
    'data_out': 'processed',
    'scale': 'scale_buffer',
    'config': 'config_buffer'
}
```

**Buffer Management:**
- `features`: Deployed
- `processed`: Deployed
- `scale_buffer`: Marked as `_deploy = False`
- `config_buffer`: Marked as `_deploy = False`

---

## Notes and Limitations

### Key Characteristics

1. **Minimal constraints:** Only upper bounds are enforced, giving the tiler maximum flexibility.

2. **No shape relationship:** Input and output shapes are not required to match. This is intentional for operations like Reshape.

3. **Auxiliary buffer management:** Automatically identifies and disables deployment for auxiliary buffers.

4. **No custom serialization:** Uses default tiling schedule generation from parent class.

### Use Cases

1. **Identity operations:** Pass-through layers that don't modify data.

2. **Reshape operations:** When the total element count is preserved but shape changes.

3. **Operators with metadata:** Operations that have auxiliary tensors (shape, indices, etc.) that are only needed at compile time.

4. **Placeholder constraints:** When a custom constraint is not yet implemented for an operator.

### Limitations

1. **No input-output relationship:** Since no equality constraints link input and output, the tiler may generate invalid tile configurations for operations that actually require shape matching.

2. **Limited use cases:** This constraint is typically a fallback or for very specific operators. Most operators need explicit shape relationships.

3. **No replacements generated:** The default serialization does not provide operator-specific replacements, which may limit code generation flexibility.

### When to Use

- Operations that truly don't require shape constraints (rare)
- Temporary placeholder while developing a proper constraint
- Operations where auxiliary buffers need to be hidden from deployment

### When NOT to Use

- Element-wise operations (use `UnaryTileConstraint` or `BOPTileConstraint`)
- Operations with specific shape relationships (create custom constraint)
- Operations that need tile-specific replacements

---

## References

- Source file: `Deeploy/Targets/Generic/TileConstraints/NOPTileConstraint.py`
- Parent class: `Deeploy/TilingExtension/TileConstraint.py`
- Related: `UntiledTileConstraint` for operations that should not be tiled at all
