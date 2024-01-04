using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace GameLogic
{
    [CreateAssetMenu(fileName = "Block", menuName = "Gamelogic/Block", order = 1)]
    public class BlockScriptableObject : ScriptableObject
    {
        /// <summary>
        /// This value is the coordinate tuple for the display of the block when it is in the list of placeable blocks.
        /// It will always be in a 5x5 grid!
        /// It follows the coordinate system of the game, which is the mathematical one (origin down left, x to right, y up)
        /// </summary>
        public Vector2Int center;

        /// <summary>
        /// List of offsets from center of the block (x,y) where origin in grid is lower left corner.
        /// positive x to the right, positive y up
        /// </summary>
        public List<Vector2Int> m_offsets;

        private bool[,] gridCache = null;

        public List<Vector2Int> getBlockOffsets()
        {
            return m_offsets;
        }
        public Vector2Int getCenter()
        {
            return center;
        }

        public bool[,] getGrid()
        {
            if (gridCache == null)
            {
                gridCache = createGrid();
            }
            return gridCache;
        }

        private bool[,] createGrid()
        {
            bool[,] grid = new bool[5, 5];
            foreach (Vector2Int offset in m_offsets)
            {
                grid[offset.x + 2, offset.y + 2] = true;
            }
            grid[2, 2] = true;
            return grid;
        }
    }
}
