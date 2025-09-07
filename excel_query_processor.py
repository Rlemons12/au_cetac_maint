#!/usr/bin/env python3
"""
Excel Query Processor Script - Production Version

This script processes part search queries from an Excel file using the actual
pattern interpreter and search handler from your system.

Requirements:
- Excel file with a 'queries' column
- Access to your database and search system

Usage: python excel_query_processor.py
"""

import pandas as pd
import os
import sys
from typing import Dict, List, Any
import logging
from datetime import datetime
import re


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import your actual system components  
# You'll need to adjust these imports based on your project structure
try:
    # Only import what we actually need
    from modules.emtacdb.emtacdb_fts import Part
    from modules.configuration.config_env import DatabaseConfig

    logger.info("Successfully imported core components")
except ImportError as e:
    logger.error(f"Failed to import system components: {e}")
    logger.error("Please ensure this script is run from your project root directory")
    sys.exit(1)


class ProductionExcelQueryProcessor:
    """Production Excel query processor using actual system components."""

    def __init__(self):
        self.db_config = DatabaseConfig()
        self.session = None
        self._setup_database_connection()

    def _setup_database_connection(self):
        """Setup database connection."""
        try:
            self.session = self.db_config.get_main_session()
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def process_file(self, file_path: str) -> str:
        """
        Process CSV or Excel file with queries and return results file path.

        Args:
            file_path: Path to CSV or Excel file with 'queries' column

        Returns:
            Path to output file with results
        """
        try:
            # Read file based on extension
            if file_path.lower().endswith('.csv'):
                logger.info(f"Reading CSV file: {file_path}")
                df = pd.read_csv(file_path)
            else:
                logger.info(f"Reading Excel file: {file_path}")
                df = pd.read_excel(file_path)

            # Validate that we have a queries column (check multiple possible names)
            query_column = None
            possible_query_columns = ['queries', 'query', 'Query', 'Queries', 'question', 'Question']

            for col_name in possible_query_columns:
                if col_name in df.columns:
                    query_column = col_name
                    break

            if query_column is None:
                available_columns = ', '.join(df.columns.tolist())
                raise ValueError(
                    f"File must contain a query column. Available columns: {available_columns}. Expected one of: {', '.join(possible_query_columns)}")

            logger.info(f"Using column '{query_column}' for queries")

            # Process each query
            results = []
            total_queries = len(df)

            logger.info(f"Processing {total_queries} queries...")

            for index, row in df.iterrows():
                query = str(row[query_column]).strip()

                if not query or query.lower() in ['nan', 'none', '']:
                    logger.info(f"Skipping empty query at row {index + 1}")
                    result = self._create_empty_result(query, index)
                else:
                    logger.info(f"Processing query {index + 1}/{total_queries}: {query}")
                    result = self._process_single_query(query, index)

                results.append(result)

            # Create output DataFrame
            output_df = self._create_output_dataframe(results)

            # Save results
            output_path = self._save_results(output_df, file_path)

            logger.info(f"Processing complete. Results saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            raise
        finally:
            if self.session:
                self.session.close()
                logger.info("Database connection closed")

    def _process_single_query(self, query: str, row_index: int) -> Dict[str, Any]:
        """Process a single query through the actual search system."""
        try:
            # Step 1: Classify intent using database patterns
            classification = self._classify_intent_from_database(query)

            if not classification:
                return {
                    'row_index': row_index,
                    'original_query': query,
                    'status': 'no_pattern_match',
                    'message': 'No matching pattern found in database',
                    'intent_name': '',
                    'pattern_matched': '',
                    'extracted_entities': {},
                    'search_params': {},
                    'part_results': [],
                    'total_results': 0,
                    'processing_time_ms': 0
                }

            # Step 2: Handle part search using classified data
            start_time = datetime.now()
            search_result = self._handle_part_search_classified(
                query,
                classification.get('extracted_data', {}),
                classification
            )
            end_time = datetime.now()
            processing_time = int((end_time - start_time).total_seconds() * 1000)

            return {
                'row_index': row_index,
                'original_query': query,
                'status': search_result.get('status', 'unknown'),
                'message': search_result.get('message', ''),
                'intent_name': classification.get('intent_name', ''),
                'pattern_matched': classification.get('pattern_text', ''),
                'extracted_entities': classification.get('extracted_data', {}),
                'search_params': search_result.get('search_params', {}),
                'part_results': search_result.get('results', []),
                'total_results': search_result.get('total_results', 0),
                'processing_time_ms': processing_time
            }

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return {
                'row_index': row_index,
                'original_query': query,
                'status': 'error',
                'message': str(e),
                'intent_name': '',
                'pattern_matched': '',
                'extracted_entities': {},
                'search_params': {},
                'part_results': [],
                'total_results': 0,
                'processing_time_ms': 0
            }

    def _classify_intent_from_database(self, question: str) -> Dict[str, Any]:
        """
        Classify user intent using database intent_pattern table.
        This is the same method from your search system.
        """
        try:
            from sqlalchemy import text

            # Query for matching patterns ordered by priority
            query = text("""
            SELECT 
                si.name as intent_name,
                si.priority as intent_priority,
                si.search_method,
                si.description,
                ip.pattern_text,
                ip.priority,
                ip.success_rate,
                ip.pattern_type,
                ip.usage_count,
                ip.id as pattern_id
            FROM intent_pattern ip
            JOIN search_intent si ON ip.intent_id = si.id
            WHERE ip.is_active = true 
              AND si.is_active = true
              AND :question ~ ip.pattern_text
            ORDER BY ip.priority DESC, ip.success_rate DESC, si.priority DESC
            LIMIT 1
            """)

            result = self.session.execute(query, {"question": question})
            row = result.fetchone()

            if row:
                # Extract data from the matched pattern
                extracted_data = self._extract_data_from_pattern(question, row.pattern_text, row.pattern_type)

                classification = {
                    'intent_name': row.intent_name,
                    'intent_priority': float(row.intent_priority),
                    'search_method': row.search_method,
                    'description': row.description,
                    'pattern_text': row.pattern_text,
                    'priority': float(row.priority),
                    'success_rate': float(row.success_rate),
                    'pattern_type': row.pattern_type,
                    'usage_count': row.usage_count,
                    'pattern_id': row.pattern_id,
                    'extracted_data': extracted_data
                }

                logger.debug(f"Intent classified: {row.intent_name} (priority: {row.priority})")
                return classification
            else:
                logger.debug("No matching intent patterns found in database")
                return None

        except Exception as e:
            logger.error(f"Error in database intent classification: {e}")
            return None

    def _extract_data_from_pattern(self, question: str, pattern_text: str, pattern_type: str) -> Dict[str, Any]:
        """Extract data from question using the matched pattern."""
        try:
            match = re.search(pattern_text, question, re.IGNORECASE)
            if match:
                groups = match.groups()

                # Create extraction data similar to your existing system
                extracted_data = {
                    'main_entity': groups[0] if groups else '',
                    'topic': groups[0] if groups else '',
                    'pattern_matched': pattern_text,
                    'pattern_type': pattern_type,
                    'extraction_confidence': 75.0,
                    'groups_captured': len(groups),
                    'full_match': match.group(0)
                }

                # Add additional groups if present
                for i, group in enumerate(groups):
                    if group:  # Only add non-empty groups
                        extracted_data[f'group_{i}'] = group

                return extracted_data

        except re.error as e:
            logger.error(f"Regex error with pattern '{pattern_text}': {e}")

        return {}

    def _handle_part_search_classified(self, question: str, extracted_data: Dict, classification: Dict) -> Dict[
        str, Any]:
        """
        Handle part searches with classification data.
        This uses your actual Part.search() method.
        """
        intent_name = classification['intent_name']
        main_entity = extracted_data.get('main_entity', '')
        pattern_text = classification.get('pattern_text', '')

        logger.info(f"Handling part search - Intent: {intent_name}, Entity: {main_entity}")

        # Build Part.search() parameters based on intent and extracted data
        search_params = self._build_search_parameters(intent_name, extracted_data, question)

        # Execute the search using your actual Part.search() method
        try:
            logger.debug(f"Executing Part.search with params: {search_params}")
            parts = Part.search(session=self.session, **search_params)

            # Convert Part objects to dictionaries for JSON serialization
            part_results = []
            for part in parts:
                part_dict = {
                    'part_number': part.part_number,
                    'name': part.name,
                    'oem_mfg': part.oem_mfg,
                    'model': part.model,
                    'class_flag': part.class_flag,
                    'type': part.type,
                    'notes': part.notes[:100] + '...' if part.notes and len(part.notes) > 100 else part.notes,
                    'documentation': part.documentation[:100] + '...' if part.documentation and len(
                        part.documentation) > 100 else part.documentation
                }
                part_results.append(part_dict)

            return {
                'status': 'success',
                'results': part_results,
                'total_results': len(part_results),
                'search_params': search_params,
                'intent_used': intent_name,
                'pattern_id': classification.get('pattern_id')
            }

        except Exception as e:
            logger.error(f"Part search failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'results': [],
                'total_results': 0,
                'search_params': search_params,
                'intent_used': intent_name
            }

    def _build_search_parameters(self, intent_name: str, extracted_data: Dict, original_question: str) -> Dict[
        str, Any]:
        """Build Part.search() parameters based on intent classification."""
        main_entity = extracted_data.get('main_entity', '')

        # Base parameters
        base_params = {
            'limit': 20,
            'use_fts': True,
            'exact_match': False
        }

        # Intent-based parameter building
        if intent_name == 'FIND_PART':
            if self._is_part_number_entity(main_entity):
                # Direct part number search
                base_params.update({
                    'part_number': main_entity
                })
                logger.debug(f"FIND_PART → Direct part number search: {main_entity}")
            else:
                # Part description search
                base_params.update({
                    'search_text': main_entity,
                    'fields': ['name', 'notes', 'documentation', 'class_flag', 'type']
                })
                logger.debug(f"FIND_PART → Description search: {main_entity}")

        elif intent_name == 'FIND_BY_MANUFACTURER':
            # Extract manufacturer and equipment from the data
            manufacturer, equipment = self._parse_manufacturer_data(extracted_data, original_question)

            base_params.update({
                'oem_mfg': manufacturer
            })

            if equipment:
                base_params.update({
                    'search_text': equipment,
                    'fields': ['name', 'notes', 'documentation', 'class_flag', 'type']
                })
                logger.debug(f"FIND_BY_MANUFACTURER → {manufacturer} + {equipment}")
            else:
                logger.debug(f"FIND_BY_MANUFACTURER → {manufacturer} only")

        elif intent_name.startswith('FIND_') and intent_name != 'FIND_PART':
            # Equipment-specific searches
            equipment_type = intent_name.replace('FIND_', '').lower()
            manufacturer = self._extract_manufacturer_from_question(original_question)

            base_params.update({
                'search_text': equipment_type,
                'fields': ['name', 'notes', 'documentation', 'class_flag', 'type']
            })

            if manufacturer:
                base_params['oem_mfg'] = manufacturer
                logger.debug(f"{intent_name} → {manufacturer} + {equipment_type}")
            else:
                logger.debug(f"{intent_name} → {equipment_type} only")

        else:
            # Fallback: general text search
            base_params.update({
                'search_text': original_question,
                'fields': ['part_number', 'name', 'oem_mfg', 'model', 'notes', 'documentation']
            })
            logger.debug(f"FALLBACK → General search: {original_question}")

        return base_params

    def _is_part_number_entity(self, entity: str) -> bool:
        """Check if extracted entity looks like a part number."""
        if not entity or len(entity) < 3:
            return False

        # Common part number patterns
        part_number_patterns = [
            r'^[A-Z]\d{5,}$',  # A138959
            r'^\d{5,}$',  # 123456  
            r'^[A-Z0-9]{2,}[-][A-Z0-9]{2,}$',  # ABC-123
            r'^[A-Z0-9]{2,}[\.][A-Z0-9]{2,}$'  # ABC.123
        ]

        entity_upper = entity.upper().strip()
        return any(re.match(pattern, entity_upper) for pattern in part_number_patterns)

    def _parse_manufacturer_data(self, extracted_data: Dict, question: str) -> tuple:
        """Parse manufacturer and equipment from extracted data."""
        main_entity = extracted_data.get('main_entity', '')

        # Handle patterns like "of banner" or "banner sensors"
        if main_entity.lower().startswith('of '):
            manufacturer = main_entity[3:].upper().strip()
            equipment = self._extract_equipment_from_question(question)
        elif ' ' in main_entity:
            # Try to split manufacturer and equipment
            parts = main_entity.split()
            manufacturer = parts[0].upper()
            equipment = ' '.join(parts[1:]).lower()
        else:
            manufacturer = main_entity.upper()
            equipment = self._extract_equipment_from_question(question)

        return manufacturer, equipment

    def _extract_equipment_from_question(self, question: str) -> str:
        """Extract equipment type from question."""
        equipment_patterns = [
            r'\b(sensors?|motors?|valves?|pumps?|bearings?|switches?|seals?|filters?|belts?|cables?)\b'
        ]

        for pattern in equipment_patterns:
            match = re.search(pattern, question.lower())
            if match:
                equipment = match.group(1)
                # Remove plural 's' for consistency
                if equipment.endswith('s') and equipment not in ['switches']:
                    equipment = equipment[:-1]
                return equipment

        return None

    def _extract_manufacturer_from_question(self, question: str) -> str:
        """Extract manufacturer name from question."""
        manufacturer_patterns = [
            r'\b(banner|siemens|omron|allen\s+bradley|schneider|abb|ge|westinghouse|rockwell|eaton)\b'
        ]

        for pattern in manufacturer_patterns:
            match = re.search(pattern, question.lower())
            if match:
                return match.group(1).upper().replace(' ', '_')

        return None

    def _create_empty_result(self, query: str, row_index: int) -> Dict[str, Any]:
        """Create empty result for invalid queries."""
        return {
            'row_index': row_index,
            'original_query': query,
            'status': 'empty_query',
            'message': 'Empty or invalid query',
            'intent_name': '',
            'pattern_matched': '',
            'extracted_entities': {},
            'search_params': {},
            'part_results': [],
            'total_results': 0,
            'processing_time_ms': 0
        }

    def _create_output_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Create output DataFrame from processing results."""
        output_data = []

        for result in results:
            # Flatten the results for Excel output
            row = {
                'row_index': result.get('row_index', ''),
                'original_query': result.get('original_query', ''),
                'status': result.get('status', ''),
                'intent_name': result.get('intent_name', ''),
                'pattern_matched': result.get('pattern_matched', ''),
                'total_results': result.get('total_results', 0),
                'processing_time_ms': result.get('processing_time_ms', 0),
                'error_message': result.get('message', ''),
                'extracted_entities': str(result.get('extracted_entities', {})),
                'search_params': str(result.get('search_params', {}))
            }

            # Add part results (first 5 parts)
            part_results = result.get('part_results', [])
            for i in range(5):  # Show up to 5 results
                if i < len(part_results):
                    part = part_results[i]
                    row[f'result_{i + 1}_part_number'] = part.get('part_number', '')
                    row[f'result_{i + 1}_name'] = part.get('name', '')
                    row[f'result_{i + 1}_manufacturer'] = part.get('oem_mfg', '')
                    row[f'result_{i + 1}_model'] = part.get('model', '')
                    row[f'result_{i + 1}_type'] = part.get('type', '')
                else:
                    row[f'result_{i + 1}_part_number'] = ''
                    row[f'result_{i + 1}_name'] = ''
                    row[f'result_{i + 1}_manufacturer'] = ''
                    row[f'result_{i + 1}_model'] = ''
                    row[f'result_{i + 1}_type'] = ''

            output_data.append(row)

        return pd.DataFrame(output_data)

    def _save_results(self, df: pd.DataFrame, original_file_path: str) -> str:
        """Save results to new Excel file."""
        # Create output filename
        base_name = os.path.splitext(os.path.basename(original_file_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_part_search_results_{timestamp}.xlsx"

        # Use same directory as input file
        output_dir = os.path.dirname(original_file_path)
        output_path = os.path.join(output_dir, output_filename)

        # Save to Excel with formatting
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Part_Search_Results', index=False)

            # Auto-adjust column widths
            worksheet = writer.sheets['Part_Search_Results']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                worksheet.column_dimensions[column_letter].width = adjusted_width

        return output_path


def main():
    """Main function to run the Excel query processor."""
    print("Production CSV/Excel Part Search Query Processor")
    print("=" * 60)
    print("This script processes part search queries from CSV or Excel using your actual database.")
    print()

    # Get file path from user
    while True:
        file_path = input("Enter the path to your CSV or Excel file (with 'query' or 'queries' column): ").strip()

        if not file_path:
            print("Please enter a file path.")
            continue

        # Remove quotes if present
        file_path = file_path.strip('"\'')

        if not os.path.exists(file_path):
            print(f" File not found: {file_path}")
            continue

        if not file_path.lower().endswith(('.xlsx', '.xls', '.csv')):
            print(" Please provide an Excel file (.xlsx or .xls) or CSV file (.csv)")
            continue

        break

    try:
        # Process the file
        print(f"\n Processing queries from: {file_path}")
        processor = ProductionExcelQueryProcessor()
        output_path = processor.process_file(file_path)

        print(f"\n Processing complete!")
        print(f" Results saved to: {output_path}")
        print(f"\nThe output file contains:")
        print("  • Original queries and processing status")
        print("  • Intent classification and pattern matches")
        print("  • Extracted entities and search parameters")
        print("  • Actual part search results from your database")
        print("  • Processing time for each query")

    except KeyboardInterrupt:
        print(f"\n Processing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n Error: {e}")
        logger.exception("Full error details:")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())