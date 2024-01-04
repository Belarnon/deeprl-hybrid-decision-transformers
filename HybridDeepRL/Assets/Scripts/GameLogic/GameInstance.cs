using System;
using System.Collections.Generic;
using System.Linq;
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
            m_blockAtlas = atlas;
            m_atlasSize = atlas.getAtlasSize();
			_setup();
        }

        public GameInstance(int width, int height, int nrGivenBlocks, BlockAtlasScriptableObject atlas) {
            m_height = height;
            m_width = width;
            m_nrGivenBlocks = nrGivenBlocks;
            m_blockAtlas = atlas;
            m_atlasSize = atlas.getAtlasSize();
            _setup();
        }

#region public

        // Fields and properties

        // Getters and setters
        public bool gameIsSolvable() { return m_isSolvable; }
        public int getScore() { return m_score; }
        public bool[,] getGrid() { return m_grid; }
        public List<BlockScriptableObject> getGivenBlocks() { return m_givenBlocks; }
        
        // Methods
        public bool putBlock(int index, Vector2Int center){
            // get the block at the index
            Debug.Assert(0 <= index && index < m_givenBlocks.Count);
            BlockScriptableObject block = m_givenBlocks[index];
            // check if the block can be set
            List<Vector2Int> offsets = block.getBlockOffsets();
            if (!_checkBlock(offsets, center)) { return false; }
            // else the block can be placed
            _setBlock(offsets, center);
            // remove given block from list
            m_givenBlocks.RemoveAt(index);
            // if the list of given blocks is empty, sample new ones
            if (m_givenBlocks.Count == 0) { _sampleNewBlocks(); }
            // check solvability
            m_isSolvable = _checkSolvability();
            // return successful setting of block
            return true;
        }
        public void reset(){
            _setup();
        }
        public void reset(BlockAtlasScriptableObject atlas){
            m_blockAtlas = atlas;
            m_atlasSize = atlas.getAtlasSize();
            _setup();
        }
        public void reset(int width, int height, int nrGivenBlocks, BlockAtlasScriptableObject atlas){
            m_height = height;
            m_width = width;
            m_nrGivenBlocks = nrGivenBlocks;
            m_blockAtlas = atlas;
            m_atlasSize = atlas.getAtlasSize();
            _setup();
        }

# endregion

#region private

        // Fields and properties
        
        //private UnityEngine.Random rand = new UnityEngine.Random();
        private int m_width, m_height, m_nrGivenBlocks, m_atlasSize;
		private bool[,] m_grid;
        private BlockAtlasScriptableObject m_blockAtlas;
        private List<BlockScriptableObject> m_givenBlocks;
        private bool m_isSolvable;
        private int m_score;

        // Methods
		private void _setup(){
            // set score to 0
            m_score = 0;
            m_isSolvable = false;
            // setup the actual grid as a 2d boolean array
            // coordinates are (width, height)
            m_grid = new bool[m_width, m_height];
            // start while loop until the created game is solvable
            while (!m_isSolvable){
                // create first nr_givenBlocks and fill it by randomly selecting
                // blocks from the atlas
                m_givenBlocks = new List<BlockScriptableObject>();
                _sampleNewBlocks();
                // check if its solvable; if not, reset the game
                m_isSolvable = _checkSolvability();
            }
		}
        private void _sampleNewBlocks(){
            for (int i=0; i<m_nrGivenBlocks; i++){
                m_givenBlocks.Add(m_blockAtlas.getBlock(UnityEngine.Random.Range(0, m_atlasSize)));
            }
        }
        private bool _checkSolvability(){
            // iterate over all free gridpoints and available blocks and check if they can be set
            // if there is a block and a center that work, return true
            foreach (BlockScriptableObject block in m_givenBlocks){
                List<Vector2Int> offsets = block.getBlockOffsets();
                for (int y=0; y < m_height; y++){
                    for (int x=0; x < m_width; x++){
                        Vector2Int center = new Vector2Int(x,y);
                        if (_checkBlock(offsets, center)) { return true; }
                    }
                }
            }
            return false;
        }
        private bool _checkBlock(List<Vector2Int> offsets, Vector2Int center){
            foreach (Vector2Int offset in offsets) {
                Vector2Int gridPoint = offset + center;
                // check if gridPoint lies on the grid!
                if (gridPoint.x < 0 || gridPoint.x >= m_width || gridPoint.y < 0 || gridPoint.y >= m_height) { return false; }
                // check if gridPoint is occupied
                if (m_grid[gridPoint.x, gridPoint.y]) { return false; }
            }
            return true;
        }
        private void _setBlock(List<Vector2Int> offsets, Vector2Int center){
            // place all gridPoints
            int scoreChange = 0;
            // meanwhile remember max/min indices for rows and cols to check for full line
            int min_width = m_width;
            int max_width = 0;
            int min_height = m_height;
            int max_height = 0;
            // set center point
            m_grid[center.x, center.y] = true;
            scoreChange++;
            foreach (Vector2Int offset in offsets){
                scoreChange++;
                Vector2Int gridPoint = offset + center;
                min_width = Math.Min(min_width, gridPoint.x);
                max_width = Math.Max(max_width, gridPoint.x);
                min_height = Math.Min(min_height, gridPoint.y);
                max_height = Math.Max(max_height, gridPoint.y);
                // change gridpoint value
                m_grid[gridPoint.x, gridPoint.y] = true;
            }
            // check grid for full lines and delete them
            scoreChange += _deleteFullLines(min_width, max_width, min_height, max_height);
            // set score
            m_score += scoreChange;
        }
        private int _deleteFullLines(int min_width, int max_width, int min_height, int max_height){
            // first check all rows and remember the full rows
            List<int> full_rows = new List<int>();
            for (int row = min_height; row <= max_height; row++){
                // check every gridpoint in this row
                bool rowIsFull = true;
                for (int i=0; i<m_width; i++){
                    // if any gridpoint in this row is not set, set bool and break the for loop
                    if (!m_grid[i, row]) { rowIsFull = false; break;}
                }
                // add row index to full_rows
                if (rowIsFull) { full_rows.Add(row); }
            }
            // then check all cols and remember the full cols
            List<int> full_cols = new List<int>();
            for (int col = min_width; col <= max_width; col++){
                // check every gridpoint in this row
                bool colIsFull = true;
                for (int i=0; i<m_height; i++){
                    // if any gridpoint in this row is not set, set bool and break the for loop
                    if (!m_grid[col, i]) { colIsFull = false; break;}
                }
                // add row index to full_rows
                if (colIsFull) { full_cols.Add(col); }
            }
            // go over every full row and delete the gridpoints
            foreach (int row in full_rows){
                for (int col=0; col<m_width; col++){
                    m_grid[col, row] = false;
                }
            }
            // go over every full col and delete the gridpoints
            foreach (int col in full_cols){
                for (int row=0; row<m_height; row++){
                    m_grid[col, row] = false;
                }
            }
            // compute score and add
            // score = (nr_cols+nr_rows) * 10 - nr_cols*nr_rows
            int deletedGridPoints = (full_rows.Count + full_cols.Count) * 10 - (full_rows.Count * full_cols.Count);
            return deletedGridPoints;
        }

#endregion
    }
}
