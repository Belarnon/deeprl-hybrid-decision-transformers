using System;
using System.Collections.Generic;
using UnityEngine;

namespace GameLogic
{

    public class GameInstance
    {
        // Constructors
        public GameInstance(BlockAtlasScriptableObject atlas) {
			m_width = 10;
			m_height = 10;
            m_nrGivenBlocks = 3;
            m_atlasSize = atlas.getAtlasSize();
			_setup();
        }

#region public

        // Fields and properties

        // Getters and setters
        
        // Methods
        public bool setBlock(BlockScriptableObject block, Vector2Int center){
            // check if the block can be set
            List<Vector2Int> offsets = block.getBlockOffsets();
            if (!_checkBlock(offsets, center)) { return false; }
            // else the block can be placed
            foreach (Vector2Int offset in offsets) {
                Vector2Int gridPoint = offset + center;
                m_grid[gridPoint.x, gridPoint.y] = true;
            }
            // return successful setting of block
            return true;
        }

        public bool checkSolvability(){
            // iterate over all free gridpoints and available blocks and check if they can be set
            // if there is a block and a center that work, return true
            foreach (BlockScriptableObject block in m_givenBlocks){
                List<Vector2Int> offsets = block.getBlockOffsets();
                for (int x=0; x < m_height; x++){
                    for (int y=0; y < m_width; y++){
                        Vector2Int center = new Vector2Int(x,y);
                        if (_checkBlock(offsets, center)) { return true; }
                    }
                }
            }
            return false;
        }

# endregion

#region private

        // Fields and properties
        
        //private UnityEngine.Random rand = new UnityEngine.Random();
        private int m_width, m_height, m_nrGivenBlocks, m_atlasSize;
		private bool[,] m_grid;
        private BlockAtlasScriptableObject m_blockAtlas;
        private BlockScriptableObject[] m_givenBlocks;
        // Methods
		private void _setup(){

			// setup the actual grid as a 2d boolean array
			m_grid = new bool[m_height, m_width];
            // create first nr_GivenBlocks and fill it by randomly selecting
            // blocks from the atlas

            m_givenBlocks = new BlockScriptableObject[m_nrGivenBlocks];
            for (int i=0; i<m_nrGivenBlocks; i++){
                m_givenBlocks[i] = m_blockAtlas.getBlock(UnityEngine.Random.Range(0, m_atlasSize));
            }
		}

        private bool _checkBlock(List<Vector2Int> offsets, Vector2Int center){
            foreach (Vector2Int offset in offsets) {
                Vector2Int gridPoint = offset + center;
                // check if gridPoint lies on the grid!
                if (gridPoint.x < 0 || gridPoint.y >= m_height || gridPoint.y < 0 || gridPoint.y >= m_width) { return false; }
                // check if gridPoint is occupied
                if (m_grid[gridPoint.x, gridPoint.y]) { return false; }
            }
            return true;
        }

#endregion
    }
}
